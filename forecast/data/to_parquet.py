"""Convert a snapshot.sh dump directory to the parquet files the rest of the project reads.

    uv run data/to_parquet.py /nfs/snapshot/2026-09-02

Writes <dir>/parquet/{station,ionosonde,ro,holdout,holdout_eval,pred_eval,runs}.parquet.
Only `station`, `ionosonde` and `ro` are cleaned; the eval tables are passed through as-is.
"""
import sys
from pathlib import Path

import duckdb

CLEAN = {
    # lat/lon are stored as text in Postgres.
    "station": """
        SELECT id, name, code, latitude::DOUBLE AS lat, longitude::DOUBLE AS lon,
               use_for_essn, use_for_maps
        FROM read_csv('{d}/station.csv.gz')""",
    "ionosonde": """
        SELECT * EXCLUDE (md), TRY_CAST(md AS DOUBLE) AS md
        FROM read_csv('{d}/measurement.csv.gz')
        ORDER BY time""",
    # One honest truth row per RO observation; model-comparison columns dropped.
    "ro": """
        SELECT DISTINCT time, latitude AS lat, longitude AS lon, fof2_true AS fof2,
               hmf2_true AS hmf2, source, dip_angle, modip
        FROM read_csv('{d}/cosmic_eval.csv.gz')
        ORDER BY time""",
}
PASSTHROUGH = ["holdout", "holdout_eval", "pred_eval", "runs", "essn"]


def main(d: Path):
    (d / "parquet").mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    for name, sql in CLEAN.items():
        out = d / "parquet" / f"{name}.parquet"
        print(name)
        con.execute(f"COPY ({sql.format(d=d)}) TO '{out}' (FORMAT parquet, COMPRESSION zstd)")
    for name in PASSTHROUGH:
        out = d / "parquet" / f"{name}.parquet"
        print(name)
        con.execute(
            f"COPY (SELECT * FROM read_csv('{d}/{name}.csv.gz')) TO '{out}' (FORMAT parquet, COMPRESSION zstd)"
        )
    for row in con.execute(
        f"SELECT file_name, num_rows FROM parquet_file_metadata('{d}/parquet/*.parquet')"
    ).fetchall():
        print(f"{row[1]:>12,}  {Path(row[0]).name}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
