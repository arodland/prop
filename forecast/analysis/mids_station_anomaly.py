"""Phase 3-style check of the archive years (PLAN.md, mids path step 2): is the foF2/hmF2 anomaly
against IRI in the old eras shaped like the one we already train on, station by station?

    uv run analysis/mids_station_anomaly.py /kass/forecast/eval/mids_ladder/y*.parquet /kass/forecast/eval/val2024_v2/all_q*.parquet

Reads the IRI rows of replay files (mode=full, so every station is a target) and joins cs back from
load.py's iono view. Each sounding is scored under up to four issue times; those collapse to one row.
Prints, per era: network anomaly stats, anomaly by cs class, and stations whose bias or spread is
far from the rest of the network in that era (robust z over stations, |z| > 3).
"""
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.load import MIDS_TRAIN, connect  # noqa: E402

ERA = "CASE WHEN year(time) < 2009 THEN '1999-2008' WHEN year(time) < 2019 THEN '2009-2018' ELSE year(time)::VARCHAR END"
MIN_N = 500  # per station-era, for the outlier table


def main(paths):
    con = connect(mids=MIDS_TRAIN)
    con.execute("SET memory_limit='24GB'")
    files = ", ".join(f"'{p}'" for p in paths)
    con.execute(f"""CREATE TEMP TABLE a AS
        SELECT r.target_id AS station_id, r.time, r.var, any_value(r.truth) AS truth, avg(r.pred) AS iri
        FROM read_parquet([{files}], union_by_name=true) r
        WHERE r.mode = 'full' AND r.model = 'iri' AND r.kind = 'iono' AND r.var IN ('fof2', 'hmf2')
        GROUP BY ALL""")
    con.execute(f"""CREATE TEMP TABLE x AS
        SELECT a.*, {ERA} AS era, truth - iri AS anom, i.cs, s.code
        FROM a JOIN iono i USING (station_id, time) JOIN station s ON s.id = a.station_id""")

    print("## Network anomaly (obs - IRI), one row per sounding")
    print(con.sql("""SELECT var, era, count(*) AS n, count(DISTINCT code) AS stations,
            round(avg(anom), 3) AS bias, round(sqrt(avg(anom^2)), 3) AS rms, round(avg(abs(anom) / iri), 3) AS rel_mae
        FROM x GROUP BY ALL ORDER BY var, era""").df().to_string(index=False))

    print("\n## By cs class (the quality filter keeps cs >= 75 and cs = -1)")
    print(con.sql("""SELECT var, era, CASE WHEN cs = -1 THEN 'unknown' WHEN cs >= 90 THEN '90-100' ELSE '75-89' END AS cs,
            count(*) AS n, round(avg(anom), 3) AS bias, round(sqrt(avg(anom^2)), 3) AS rms
        FROM x GROUP BY ALL ORDER BY var, era, cs""").df().to_string(index=False))

    print(f"\n## Outlier station-eras (|robust z| > 3 on bias or rms among stations of that era; n >= {MIN_N})")
    print(con.sql(f"""WITH st AS (
            SELECT var, era, code, count(*) AS n, avg(anom) AS bias, sqrt(avg(anom^2)) AS rms, avg((cs = -1)::INT) AS cs_unknown
            FROM x GROUP BY ALL HAVING count(*) >= {MIN_N}),
        z AS (SELECT *, (bias - median(bias) OVER w) / (1.4826 * mad(bias) OVER w) AS z_bias,
                        (rms - median(rms) OVER w) / (1.4826 * mad(rms) OVER w) AS z_rms, count(*) OVER w AS peers
              FROM st WINDOW w AS (PARTITION BY var, era))
        SELECT var, era, code, n, round(bias, 2) AS bias, round(rms, 2) AS rms, round(z_bias, 1) AS z_bias,
               round(z_rms, 1) AS z_rms, round(cs_unknown, 2) AS cs_unknown
        FROM z WHERE peers >= 5 AND (abs(z_bias) > 3 OR abs(z_rms) > 3) ORDER BY var, era, greatest(abs(z_bias), abs(z_rms)) DESC""").df().to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
