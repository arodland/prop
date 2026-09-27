"""Read a snapshot's parquet into the frames everything downstream uses.

Physical bounds and the quality rule live here and only here (PLAN.md Phase 1: fixed once, never tuned).

connect(mids=(start, end)) adds the NOAA archive (data/mids_to_duckdb.py) for soundings in
[start, end) that the snapshot does not already have. It is off by default so every existing number
reproduces; MIDS_TRAIN stops before 2024 so the 2024 validation slice stays snapshot-only.
"""
from pathlib import Path

import duckdb

SNAPSHOT = Path("/kass/forecast/snapshot/2026-09-02/parquet")
MIDS = Path("/kass/forecast/mids.duckdb")
MIDS_TRAIN = ("1999-01-01", "2024-01-01")
MIDS_ID_BASE = 100_000  # station ids for archive codes the snapshot has no station for; int32-safe
DUP_S = 60  # an archive row within this many seconds of a snapshot row at the same code is the same sounding

# Quality: cs >= 75, or unknown (-1). Physical bounds: outside these is autoscaler garbage.
QUALITY = "(cs >= 75 OR cs = -1)"
BOUNDS = {
    "fof2": (0.5, 25.0),   # MHz
    "hmf2": (100.0, 600.0),  # km
    "mufd": (1.0, 80.0),   # MHz
    "tec": (0.0, 200.0),   # TECU
    "foe": (0.1, 10.0),    # MHz
}
CLUSTER_KM = 50  # co-located URSI codes (Rome x3, Tromso x2, ...) share a cluster id


def _null_out_of_bounds():
    return ", ".join(
        f"CASE WHEN {v} BETWEEN {lo} AND {hi} THEN {v} END AS {v}" for v, (lo, hi) in BOUNDS.items()
    )


def mids_extra(snapshot: Path = SNAPSHOT, mids: Path = MIDS) -> Path:
    """The archive's contribution to `snapshot`, cached as <mids dir>/mids-extra/<snapshot>/.

    ionosonde.parquet: archive rows with no snapshot row at the same code within DUP_S seconds,
    carrying the snapshot's station_id where the code exists there and MIDS_ID_BASE+ otherwise.
    station.parquet: stations for those new codes, in the snapshot's station schema.

    Built once (a couple of minutes) and rebuilt when mids.duckdb or the snapshot is newer. Reading
    mids.duckdb needs no other process to hold it open for writing: use `duckdb -readonly`.
    """
    out = mids.parent / "mids-extra" / snapshot.parent.name
    sources = [mids, snapshot / "ionosonde.parquet", snapshot / "station.parquet"]
    if (out / "ionosonde.parquet").exists() and all(
            (out / "ionosonde.parquet").stat().st_mtime > s.stat().st_mtime for s in sources):
        return out
    tmp = out.with_name(out.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB'; SET threads = 8")
    con.execute(f"ATTACH '{mids}' AS m (READ_ONLY)")
    con.execute(f"""
        CREATE TEMP TABLE ids AS
        SELECT code, id FROM '{snapshot}/station.parquet'
        UNION ALL
        SELECT code, {MIDS_ID_BASE} + row_number() OVER (ORDER BY code) AS id
        FROM m.station WHERE code NOT IN (SELECT code FROM '{snapshot}/station.parquet');
        -- snapshot soundings by code and minute, spread one minute either way for the join below
        CREATE TEMP TABLE snap AS
        SELECT DISTINCT s.code, i.time, (epoch(i.time) // 60)::BIGINT + d AS k
        FROM '{snapshot}/ionosonde.parquet' i JOIN '{snapshot}/station.parquet' s ON s.id = i.station_id,
             (VALUES (-1), (0), (1)) v(d);
        CREATE TEMP TABLE covered AS
        SELECT DISTINCT a.code, a.time
        FROM m.ionosonde a JOIN snap b ON b.code = a.code AND b.k = (epoch(a.time) // 60)::BIGINT
        WHERE abs(epoch(a.time) - epoch(b.time)) <= {DUP_S};
        COPY (SELECT ids.id AS station_id, a.* FROM m.ionosonde a JOIN ids USING (code)
              ANTI JOIN covered USING (code, time) ORDER BY time)
          TO '{tmp}/ionosonde.parquet' (FORMAT parquet, COMPRESSION zstd);
        COPY (SELECT ids.id, st.name, st.code, st.lat, st.lon, false AS use_for_essn, false AS use_for_maps
              FROM m.station st JOIN ids USING (code) WHERE ids.id >= {MIDS_ID_BASE} ORDER BY ids.id)
          TO '{tmp}/station.parquet' (FORMAT parquet);
    """)
    con.close()
    if out.exists():
        for f in out.iterdir():
            f.unlink()
        out.rmdir()
    tmp.rename(out)
    return out


def connect(snapshot: Path = SNAPSHOT, mids: tuple[str, str] | None = None) -> duckdb.DuckDBPyConnection:
    """Views: station (with cluster), iono (quality-filtered, bounds-nulled), ro.

    mids=(start, end) adds archive soundings in [start, end) that the snapshot lacks (mids_extra);
    None, the default, is the snapshot alone.
    """
    con = duckdb.connect()
    station_src = f"SELECT * FROM '{snapshot}/station.parquet'"
    iono_src = f"SELECT id, station_id, time, cs, source, {_null_out_of_bounds()} FROM '{snapshot}/ionosonde.parquet' WHERE {QUALITY}"
    if mids is not None:
        extra = mids_extra(snapshot)
        start, end = mids
        station_src += f" UNION ALL SELECT * FROM '{extra}/station.parquet'"
        iono_src += f"""
            UNION ALL
            SELECT NULL::BIGINT AS id, station_id, time, cs, source, {_null_out_of_bounds()}
            FROM '{extra}/ionosonde.parquet'
            WHERE {QUALITY} AND time >= TIMESTAMP '{start}' AND time < TIMESTAMP '{end}'"""
    con.execute(f"""
        CREATE VIEW station_raw AS {station_src};
        CREATE VIEW station AS
        WITH pairs AS (
            SELECT a.id, min(b.id) AS cluster
            FROM station_raw a JOIN station_raw b
              ON 6371 * acos(least(1.0, sin(radians(a.lat))*sin(radians(b.lat))
                 + cos(radians(a.lat))*cos(radians(b.lat))*cos(radians(a.lon-b.lon)))) < {CLUSTER_KM}
            GROUP BY a.id)
        SELECT s.*, p.cluster FROM station_raw s JOIN pairs p USING (id);
        CREATE VIEW iono AS {iono_src};
        CREATE VIEW ro AS
        SELECT * FROM '{snapshot}/ro_full.parquet' WHERE NOT isnan(fof2) AND fof2 BETWEEN 0.5 AND 25;  -- continuous UCAR re-derivation (2019-10 ->)
    """)
    return con


def inputs_at(con, t, hours=48, latency_min=15):
    """Point-in-time observation set: rows with time in [t-hours, t-latency]. Returns a relation."""
    return con.sql(f"""
        SELECT i.*, s.lat, s.lon, s.cluster FROM iono i JOIN station s ON s.id = i.station_id
        WHERE i.time > TIMESTAMP '{t}' - INTERVAL {hours} HOUR
          AND i.time <= TIMESTAMP '{t}' - INTERVAL {latency_min} MINUTE""")


def targets_after(con, t, hours=24):
    """Ionosonde and RO truth in (t, t+hours]."""
    iono = con.sql(f"""
        SELECT 'iono' AS kind, i.station_id AS target_id, s.cluster, i.time, s.lat, s.lon, i.fof2, i.hmf2, i.mufd, i.tec
        FROM iono i JOIN station s ON s.id = i.station_id
        WHERE i.time > TIMESTAMP '{t}' AND i.time <= TIMESTAMP '{t}' + INTERVAL {hours} HOUR""")
    ro = con.sql(f"""
        SELECT 'ro' AS kind, NULL::BIGINT AS target_id, NULL::BIGINT AS cluster, time, lat, lon, fof2, hmf2,
               NULL::DOUBLE AS mufd, NULL::DOUBLE AS tec
        FROM ro WHERE time > TIMESTAMP '{t}' AND time <= TIMESTAMP '{t}' + INTERVAL {hours} HOUR""")
    return iono.union(ro)


if __name__ == "__main__":
    con = connect()
    n_clusters, n_stations = con.execute("SELECT count(DISTINCT cluster), count(*) FROM station").fetchone()
    assert n_clusters < n_stations, "co-located stations should collapse"
    rome = con.execute("SELECT count(DISTINCT cluster) FROM station WHERE code IN ('RM041','RO041','RA041')").fetchone()[0]
    assert rome == 1
    t = "2024-03-15 12:00:00"
    x = inputs_at(con, t).aggregate("count(*), max(time), min(time)").fetchone()
    y = targets_after(con, t).aggregate("count(*), min(time), max(time), count(fof2)").fetchone()
    assert x[0] > 0 and str(x[1]) <= "2024-03-15 11:45:00"
    assert y[0] > 0 and str(y[1]) > t and str(y[2]) <= "2024-03-16 12:00:00"
    bad = con.execute("SELECT count(*) FROM iono WHERE fof2 > 25 OR hmf2 > 600").fetchone()[0]
    assert bad == 0
    print(f"ok: {n_stations} stations in {n_clusters} clusters; inputs@{t}: {x[0]} rows; targets: {y[0]} rows")

    # With the archive: new stations join without moving any existing station's cluster, the
    # 2024 validation slice is untouched, and older issue times gain inputs.
    m = connect(mids=MIDS_TRAIN)
    n_all = m.execute("SELECT count(*) FROM station").fetchone()[0]
    old = dict(con.execute("SELECT id, cluster FROM station").fetchall())
    new = dict(m.execute("SELECT id, cluster FROM station").fetchall())
    assert n_all > n_stations and all(new[i] == c for i, c in old.items()), "existing clusters must not move"
    sl, rl = (m.execute(f"SELECT cluster FROM station WHERE code = '{c}'").fetchone()[0] for c in ("SL051", "RL052"))
    assert sl == rl, "SL051 is the Chilton site's old code and must share RL052's cluster"
    q24 = "SELECT count(*) FROM iono WHERE time >= '2024-01-01' AND time < '2025-01-01'"
    assert m.execute(q24).fetchone() == con.execute(q24).fetchone(), "2024 must stay snapshot-only"
    t0 = "2010-06-15 12:00:00"
    a, b = (inputs_at(c, t0).aggregate("count(*)").fetchone()[0] for c in (con, m))
    assert b > a, (a, b)
    # No station gains a second row within DUP_S of another on a busy overlap day.
    near = """SELECT count(*) FROM iono x JOIN iono y ON x.station_id = y.station_id AND x.time < y.time
              AND epoch(y.time) - epoch(x.time) <= 60 WHERE x.time::DATE = '2020-03-15' AND y.time::DATE = '2020-03-15'"""
    assert m.execute(near).fetchone() == con.execute(near).fetchone(), "archive rows duplicated snapshot rows"
    print(f"ok with archive: {n_all} stations; inputs@{t0}: {a} -> {b} rows; 2024 slice unchanged")
