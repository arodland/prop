"""Print a data card (markdown) for a snapshot parquet directory.

    uv run data/card.py /kass/forecast/snapshot/2026-09-02/parquet > DATA_CARD.md
"""
import sys

import duckdb


def table(con, title, sql):
    cur = con.execute(sql)
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()
    print(f"\n### {title}\n")
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(f"{v:,}" if isinstance(v, int) and abs(v) >= 10000 else str(v) if isinstance(v, int) else f"{v:.3g}" if isinstance(v, float) else str(v) for v in r) + " |")


def main(d):
    con = duckdb.connect()
    con.execute(f"CREATE VIEW iono AS SELECT * FROM '{d}/ionosonde.parquet'")
    con.execute(f"CREATE VIEW ro AS SELECT * FROM '{d}/ro.parquet'")
    con.execute(f"CREATE VIEW station AS SELECT * FROM '{d}/station.parquet'")
    print(f"# Data card: {d}")

    table(con, "Ionosonde rows by year", """
        SELECT year(time) AS year, count(*) AS rows, count(DISTINCT station_id) AS stations,
               round(avg((fof2 IS NOT NULL)::INT),3) AS has_fof2, round(avg((hmf2 IS NOT NULL)::INT),3) AS has_hmf2,
               round(avg((mufd IS NOT NULL)::INT),3) AS has_mufd, round(avg((tec IS NOT NULL)::INT),3) AS has_tec,
               round(avg((cs >= 75 OR cs = -1)::INT),3) AS pass_cs
        FROM iono GROUP BY 1 ORDER BY 1""")
    table(con, "Ionosonde rows by source", """
        SELECT source, count(*) AS rows, min(time)::DATE AS first, max(time)::DATE AS last,
               round(avg((cs = -1)::INT),3) AS cs_unknown, round(avg((cs >= 75)::INT),3) AS cs_ge75
        FROM iono GROUP BY 1 ORDER BY 2 DESC""")
    table(con, "Confidence score distribution", """
        SELECT CASE WHEN cs = -1 THEN 'unknown(-1)' WHEN cs < 50 THEN '<50' WHEN cs < 75 THEN '50-74'
                    WHEN cs < 90 THEN '75-89' ELSE '90-100' END AS cs_bin, count(*) AS rows,
               round(avg(fof2),2) AS mean_fof2
        FROM iono GROUP BY 1 ORDER BY 1""")
    table(con, "Value ranges (cs-passing rows)", """
        SELECT 'fof2' AS var, count(fof2) AS n, min(fof2) AS min, quantile_cont(fof2,0.01) AS p1,
               median(fof2) AS med, quantile_cont(fof2,0.99) AS p99, max(fof2) AS max FROM iono WHERE cs>=75 OR cs=-1
        UNION ALL SELECT 'hmf2', count(hmf2), min(hmf2), quantile_cont(hmf2,0.01), median(hmf2), quantile_cont(hmf2,0.99), max(hmf2) FROM iono WHERE cs>=75 OR cs=-1
        UNION ALL SELECT 'mufd', count(mufd), min(mufd), quantile_cont(mufd,0.01), median(mufd), quantile_cont(mufd,0.99), max(mufd) FROM iono WHERE cs>=75 OR cs=-1
        UNION ALL SELECT 'tec', count(tec), min(tec), quantile_cont(tec,0.01), median(tec), quantile_cont(tec,0.99), max(tec) FROM iono WHERE cs>=75 OR cs=-1
        UNION ALL SELECT 'foe', count(foe), min(foe), quantile_cont(foe,0.01), median(foe), quantile_cont(foe,0.99), max(foe) FROM iono WHERE cs>=75 OR cs=-1""")
    table(con, "Stations: uptime over 2019-2026 (fraction of hours with a cs-passing row)", """
        WITH h AS (SELECT station_id, count(DISTINCT date_trunc('hour', time)) AS hours
                   FROM iono WHERE time >= '2019-01-01' AND (cs>=75 OR cs=-1) GROUP BY 1),
             span AS (SELECT datediff('hour', TIMESTAMP '2019-01-01', (SELECT max(time) FROM iono)) AS total)
        SELECT s.id, s.code, s.name, round(s.lat,1) AS lat, round(s.lon,1) AS lon, s.use_for_maps,
               h.hours, round(h.hours / span.total, 3) AS uptime
        FROM station s LEFT JOIN h ON h.station_id = s.id, span ORDER BY uptime DESC NULLS LAST""")
    table(con, "Cadence: median minutes between consecutive rows per station (2024)", """
        WITH g AS (SELECT station_id, datediff('minute', lag(time) OVER (PARTITION BY station_id ORDER BY time), time) AS gap
                   FROM iono WHERE year(time)=2024)
        SELECT CASE WHEN m <= 5 THEN '<=5' WHEN m <= 10 THEN '6-10' WHEN m <= 15 THEN '11-15' WHEN m <= 30 THEN '16-30' WHEN m <= 60 THEN '31-60' ELSE '>60' END AS median_gap_min,
               count(*) AS stations
        FROM (SELECT station_id, median(gap) AS m FROM g WHERE gap IS NOT NULL GROUP BY 1) GROUP BY 1 ORDER BY min(m)""")
    table(con, "Duplicate (station, time) pairs", """
        SELECT count(*) AS dup_pairs FROM (SELECT station_id, time FROM iono GROUP BY 1,2 HAVING count(*) > 1)""")

    table(con, "RO observations by year and source", """
        SELECT year(time) AS year, source, count(*) AS rows, round(avg(fof2),2) AS mean_fof2,
               round(avg(hmf2),1) AS mean_hmf2, round(avg((abs(lat) < 30)::INT),3) AS frac_lowlat
        FROM ro GROUP BY 1,2 ORDER BY 1,2""")
    table(con, "RO duplicate check: distinct (time,lat,lon) vs rows", """
        SELECT count(*) AS rows, count(DISTINCT (time, lat, lon)) AS distinct_obs FROM ro""")
    table(con, "RO value ranges", """
        SELECT 'fof2' AS var, min(fof2) AS min, quantile_cont(fof2,0.01) AS p1, median(fof2) AS med, quantile_cont(fof2,0.99) AS p99, max(fof2) AS max FROM ro
        UNION ALL SELECT 'hmf2', min(hmf2), quantile_cont(hmf2,0.01), median(hmf2), quantile_cont(hmf2,0.99), max(hmf2) FROM ro""")

    table(con, "Split sizes (PLAN.md): cs-passing ionosonde rows and RO rows", """
        SELECT split, sum(iono) AS ionosonde, sum(ro) AS ro FROM (
          SELECT CASE WHEN time < '2024-01-01' THEN 'train (<2024)' WHEN time < '2025-01-01' THEN 'val (2024)'
                      WHEN time < '2026-07-01' THEN 'test (2025-01..2026-06)' ELSE 'live (2026-07+)' END AS split,
                 1 AS iono, 0 AS ro FROM iono WHERE cs>=75 OR cs=-1
          UNION ALL
          SELECT CASE WHEN time < '2024-01-01' THEN 'train (<2024)' WHEN time < '2025-01-01' THEN 'val (2024)'
                      WHEN time < '2026-07-01' THEN 'test (2025-01..2026-06)' ELSE 'live (2026-07+)' END, 0, 1 FROM ro
        ) GROUP BY 1 ORDER BY min(CASE split WHEN 'train (<2024)' THEN 0 WHEN 'val (2024)' THEN 1 WHEN 'test (2025-01..2026-06)' THEN 2 ELSE 3 END)""")


if __name__ == "__main__":
    main(sys.argv[1])
