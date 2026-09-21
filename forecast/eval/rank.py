"""One line per model: RMSE by lead bucket (holdout ionosonde), plus RO and full-mode RMSE, sorted by primary.

    uv run eval/rank.py /kass/forecast/eval/val2024/all_q*.parquet [--var fof2]
"""
import argparse

import duckdb

from report import BUCKET_SQL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--var", default="fof2")
    a = ap.parse_args()
    con = duckdb.connect()
    con.execute("SET memory_limit='32GB'")
    files = ", ".join(f"'{p}'" for p in a.paths)
    con.execute(f"""CREATE VIEW rows AS SELECT model, mode, kind, lead_h, truth, pred, issue_time, {BUCKET_SQL} AS bucket
        FROM read_parquet([{files}], union_by_name=true)
        WHERE var = '{a.var}' AND lead_h <= 24 AND pred IS NOT NULL AND NOT isnan(pred)""")
    ho = con.execute("""SELECT model, bucket, sqrt(avg((pred-truth)^2)) AS rmse FROM rows
        WHERE mode='holdout' AND kind='iono' GROUP BY 1,2""").df().pivot(index="model", columns="bucket", values="rmse")
    ho = ho[[c for c in ("0-1h", "1-3h", "3-6h", "6-12h", "12-24h") if c in ho.columns]]
    ho["primary"] = ho.mean(axis=1)
    extra = con.execute("""SELECT model,
        sqrt(avg(CASE WHEN mode='full' AND kind='ro' THEN (pred-truth)^2 END)) AS ro,
        sqrt(avg(CASE WHEN mode='full' AND kind='iono' THEN (pred-truth)^2 END)) AS "full"
        FROM rows GROUP BY 1""").df().set_index("model")
    t = ho.join(extra, how="outer")
    meta = con.execute("SELECT count(DISTINCT issue_time), count(*) FILTER (WHERE mode='holdout' AND kind='iono') / count(DISTINCT model) FROM rows").fetchone()
    print(f"{a.var}: issue times {meta[0]}, holdout rows/model ~{int(meta[1]):,}")
    print(t.sort_values("primary").round(3).to_string())


if __name__ == "__main__":
    main()
