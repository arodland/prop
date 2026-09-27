"""Build the working DuckDB copy of the mids archive from its parquet tree.

    uv run data/mids_to_duckdb.py [/kass/forecast/mids-parquet] [/kass/forecast/mids.duckdb]

The parquet tree stays as the instruments reported it, for export. This copy is what training
reads, and differs from it in two ways:

- Rows known to be another station's soundings are left out (EXCLUDE; the evidence is in
  MIDS_ANOMALIES.md). Rows with corrupted coordinates but the right station stay in.
- Missing M(D) and MUF(D) are filled the way loader/app/load.pl does, and flagged:
  MUF(D) = foF2 * M(D) when M(D) is there; otherwise M(D) is taken from hmF2 over a 3000 km
  path and MUF(D) = foF2 * M(D). md_derived / mufd_derived say which values were filled.

Tables: ionosonde (one row per sounding, the parquet schema plus the two flags), station (from
station.parquet), excluded (row counts per EXCLUDE rule). Built into <db>.tmp and renamed.
"""
import sys
from pathlib import Path

import duckdb

# (label, SQL predicate on the parquet rows). Each is a code-and-period whose rows carry another
# station's soundings; see MIDS_ANOMALIES.md for the evidence.
EXCLUDE = [
    ("GR13L holding CO764 (College) 1999-2003", "code = 'GR13L' AND lat = 64.9"),
    ("FF051 holding WP937 (Wallops) 2009", "code = 'FF051' AND lat = 37.9 AND lon = 284.5"),
    ("CW46O holding AT138 (Athens) 2008", "code = 'CW46O' AND lat = 38.0 AND lon = 23.5"),
    ("CW46O holding CN53L 2008", "code = 'CW46O' AND lat = -34.05 AND lon = 150.67"),
    ("OS768 is SN437", "code = 'OS768'"),
]
# (label, code, other): same site as the station they repeat, so only the rows with a sounding of
# `other` within a minute are dropped; the rest fill gaps.
REPEATS = [
    ("SL051 repeating RL052 (Chilton)", "SL051", "RL052"),
    ("ND51R repeating ND61R (Niue)", "ND51R", "ND61R"),
]

# load.pl: M(D) for a 3000 km path from hmF2, with the layer's base 37 km below the peak.
MD_FROM_HMF2 = "1 / cos(atan2(sin(1500 / 6371.0), 1 + (hmf2 + 37) / 6371.0 - cos(1500 / 6371.0)))"


def main(src: Path, db: Path):
    tmp = db.with_suffix(".duckdb.tmp")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET threads = 8")
    con.execute(f"CREATE TEMP VIEW raw AS SELECT * FROM read_parquet('{src}/year=*/*.parquet')")

    # (code, time) is unique in the tree, so the rows to leave out are collected as keys first.
    con.execute("CREATE TEMP TABLE drop_keys (rule VARCHAR, code VARCHAR, time TIMESTAMP)")
    for label, where in EXCLUDE:
        con.execute(f"INSERT INTO drop_keys SELECT ?, code, time FROM raw WHERE {where}", [label])
    for label, code, other in REPEATS:
        con.execute("""
            INSERT INTO drop_keys
            SELECT DISTINCT ?, a.code, a.time
            FROM (SELECT code, time FROM raw WHERE code = ?) a
            JOIN (SELECT time FROM raw WHERE code = ?) b
              ON b.time BETWEEN a.time - INTERVAL 1 MINUTE AND a.time + INTERVAL 1 MINUTE""",
                    [label, code, other])
    con.execute("CREATE TABLE excluded AS SELECT rule, count(*) AS rows FROM drop_keys GROUP BY rule")

    cols = [r[0] for r in con.execute("DESCRIBE raw").fetchall()]
    passthrough = ", ".join(c for c in cols if c not in ("md", "mufd"))
    con.execute(f"""
        CREATE TABLE ionosonde AS
        WITH k AS (SELECT * FROM raw ANTI JOIN drop_keys USING (code, time)),
        f AS (SELECT *,
                 CASE WHEN md IS NULL AND mufd IS NULL AND fof2 IS NOT NULL AND hmf2 IS NOT NULL
                      THEN {MD_FROM_HMF2} END AS md_calc
              FROM k)
        SELECT {passthrough},
               coalesce(md, md_calc) AS md,
               coalesce(mufd, fof2 * coalesce(md, md_calc)) AS mufd,
               md IS NULL AND md_calc IS NOT NULL AS md_derived,
               mufd IS NULL AND fof2 IS NOT NULL AND coalesce(md, md_calc) IS NOT NULL AS mufd_derived
        FROM f ORDER BY time, code""")
    con.execute(f"CREATE TABLE station AS SELECT * FROM '{src}/station.parquet'")
    con.close()
    tmp.replace(db)

    con = duckdb.connect(str(db), read_only=True)
    print(con.execute("SELECT * FROM excluded").fetchdf().to_string(index=False))
    print(con.execute("""
        SELECT count(*) AS rows, count(md) AS md, count(*) FILTER (WHERE md_derived) AS md_derived,
               count(mufd) AS mufd, count(*) FILTER (WHERE mufd_derived) AS mufd_derived
        FROM ionosonde""").fetchdf().to_string(index=False))


def _selfcheck():
    """The fill formulas against load.pl's, on hand-made rows."""
    import math
    con = duckdb.connect()
    con.execute("CREATE TABLE t (fof2 DOUBLE, md DOUBLE, mufd DOUBLE, hmf2 DOUBLE)")
    con.execute("INSERT INTO t VALUES (6.0, 3.0, NULL, 300), (6.0, NULL, NULL, 300), (6.0, NULL, 20.0, 300),"
                " (NULL, 3.0, NULL, 300), (6.0, NULL, NULL, NULL)")
    got = con.execute(f"""SELECT coalesce(md, md_calc), coalesce(mufd, fof2 * coalesce(md, md_calc)) FROM
        (SELECT *, CASE WHEN md IS NULL AND mufd IS NULL AND fof2 IS NOT NULL AND hmf2 IS NOT NULL
                   THEN {MD_FROM_HMF2} END AS md_calc FROM t)""").fetchall()
    md300 = 1 / math.cos(math.atan2(math.sin(1500 / 6371), 1 + (300 + 37) / 6371 - math.cos(1500 / 6371)))
    assert got[0] == (3.0, 18.0)                        # M(D) present: MUF = foF2 * M(D)
    assert got[1] == (md300, 6.0 * md300)               # from hmF2
    assert got[2] == (None, 20.0)                       # MUF present: nothing filled
    assert got[3] == (3.0, None)                        # no foF2: nothing filled
    assert got[4] == (None, None)                       # no M(D), no hmF2: nothing filled
    assert 2.9 < md300 < 3.4, md300                     # M(3000)F2 is ~3 for a 300 km peak
    print("selfcheck ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["selfcheck"]:
        _selfcheck()
    else:
        main(Path(sys.argv[1] if len(sys.argv) > 1 else "/kass/forecast/mids-parquet"),
             Path(sys.argv[2] if len(sys.argv) > 2 else "/kass/forecast/mids.duckdb"))
