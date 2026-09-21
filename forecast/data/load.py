"""Read a snapshot's parquet into the frames everything downstream uses.

Physical bounds and the quality rule live here and only here (PLAN.md Phase 1: fixed once, never tuned).
"""
from pathlib import Path

import duckdb

SNAPSHOT = Path("/kass/forecast/snapshot/2026-09-02/parquet")

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


def connect(snapshot: Path = SNAPSHOT) -> duckdb.DuckDBPyConnection:
    """Views: station (with cluster), iono (quality-filtered, bounds-nulled), ro."""
    con = duckdb.connect()
    con.execute(f"""
        CREATE VIEW station_raw AS SELECT * FROM '{snapshot}/station.parquet';
        CREATE VIEW station AS
        WITH pairs AS (
            SELECT a.id, min(b.id) AS cluster
            FROM station_raw a JOIN station_raw b
              ON 6371 * acos(least(1.0, sin(radians(a.lat))*sin(radians(b.lat))
                 + cos(radians(a.lat))*cos(radians(b.lat))*cos(radians(a.lon-b.lon)))) < {CLUSTER_KM}
            GROUP BY a.id)
        SELECT s.*, p.cluster FROM station_raw s JOIN pairs p USING (id);
        CREATE VIEW iono AS
        SELECT id, station_id, time, cs, source, {_null_out_of_bounds()}
        FROM '{snapshot}/ionosonde.parquet' WHERE {QUALITY};
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
