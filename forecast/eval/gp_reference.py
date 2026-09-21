"""Convert production GP holdout forecasts (pred_eval, control branches) into the replay schema.

    uv run eval/gp_reference.py /kass/forecast/eval/gp2023

Writes gp_ref.parquet (models gp_prod / iri_prod / irimap_prod, mode=holdout, kind=iono) and
holdouts.parquet (issue_time, station_id) so replay.py --holdouts scores our models on the same
issue times and withheld stations. Only runs on the hour are kept.
"""
import sys
from pathlib import Path

import duckdb

SNAP = "/kass/forecast/snapshot/2026-09-02/parquet"
EXPERIMENTS = ("2023-05b-ipe-control", "2023-07-control")
MODEL_NAMES = {"assimilated": "gp_prod", "iri": "iri_prod", "irimap": "irimap_prod"}


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    exps = ", ".join(f"'{e}'" for e in EXPERIMENTS)
    con.execute(f"""
        CREATE TABLE ref AS
        SELECT r.target_time AS issue_time, 'holdout' AS mode,
               CASE p.model {' '.join(f"WHEN '{k}' THEN '{v}'" for k, v in MODEL_NAMES.items())} END AS model,
               'iono' AS kind, h.station_id AS target_id, s.id AS cluster, m.time,
               (epoch(m.time) - epoch(r.target_time)) / 3600.0 AS lead_h, s.lat, s.lon,
               p.fof2 AS pred_fof2, p.hmf2 AS pred_hmf2, p.mufd AS pred_mufd,
               m.fof2 AS truth_fof2, m.hmf2 AS truth_hmf2, m.mufd AS truth_mufd
        FROM '{SNAP}/pred_eval.parquet' p
        JOIN '{SNAP}/holdout.parquet' h ON h.id = p.holdout_id
        JOIN '{SNAP}/runs.parquet' r ON r.id = h.run_id
        JOIN '{SNAP}/ionosonde.parquet' m ON m.id = p.measurement_id
        JOIN '{SNAP}/station.parquet' s ON s.id = h.station_id
        WHERE r.experiment IN ({exps}) AND p.model IN ({', '.join(f"'{k}'" for k in MODEL_NAMES)})
          AND extract(minute FROM r.target_time) = 0 AND (m.cs >= 75 OR m.cs = -1)
          AND m.time > r.target_time AND m.time <= r.target_time + INTERVAL 24 HOUR""")
    con.execute(f"""
        COPY (
          SELECT issue_time, mode, model, kind, target_id, cluster, time, lead_h, lat, lon, var, truth, pred FROM (
            SELECT *, 'fof2' AS var, truth_fof2 AS truth, pred_fof2 AS pred FROM ref
            UNION ALL SELECT *, 'hmf2', truth_hmf2, pred_hmf2 FROM ref
            UNION ALL SELECT *, 'mufd', truth_mufd, pred_mufd FROM ref)
          WHERE truth IS NOT NULL AND pred IS NOT NULL
        ) TO '{out}/gp_ref.parquet' (FORMAT parquet)""")
    con.execute(f"""
        COPY (SELECT DISTINCT issue_time, target_id AS station_id FROM ref ORDER BY issue_time)
        TO '{out}/holdouts.parquet' (FORMAT parquet)""")
    print(con.execute(f"SELECT model, count(*) FROM '{out}/gp_ref.parquet' GROUP BY 1").fetchall())
    print(con.execute(f"SELECT count(*), count(DISTINCT issue_time), min(issue_time), max(issue_time) FROM '{out}/holdouts.parquet'").fetchall())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
