"""Skill tables from replay parquet: RMSE by lead bucket, skill vs a reference, block-bootstrap CI over issue days.

    uv run eval/report.py /kass/forecast/eval/val2024/all_q*.parquet [--mode holdout] [--kind iono] [--ref iri]

All heavy work is in duckdb with filter pushdown; only per-day sums reach Python.
"""
import argparse

import duckdb
import numpy as np

BUCKETS = [(0, 1), (1, 3), (3, 6), (6, 12), (12, 24)]
BUCKET_SQL = "CASE " + " ".join(f"WHEN lead_h <= {b} THEN '{a}-{b}h'" for a, b in BUCKETS) + " END"


def connect(paths, mode, kind, var, memory="32GB", start=None, end=None):
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory}'")
    files = ", ".join(f"'{p}'" for p in paths)
    cols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet([{files}], union_by_name=true)").fetchall()}
    if "sigma" not in cols:
        con.execute(f"CREATE VIEW _src AS SELECT *, NULL::DOUBLE AS sigma FROM read_parquet([{files}], union_by_name=true)")
    else:
        con.execute(f"CREATE VIEW _src AS SELECT * FROM read_parquet([{files}], union_by_name=true)")
    con.execute(f"""CREATE VIEW rows AS
        SELECT issue_time, model, target_id, time, lat, lon, lead_h, truth, pred, sigma, {BUCKET_SQL} AS bucket
        FROM _src
        WHERE mode = '{mode}' AND kind = '{kind}' AND var = '{var}' AND lead_h <= 24 AND pred IS NOT NULL AND NOT isnan(pred)
          {"AND issue_time >= '" + start + "'" if start else ""} {"AND issue_time < '" + end + "'" if end else ""}""")
    return con


def rmse_by_bucket(con):
    t = con.execute("""
        SELECT model, bucket, sqrt(avg((pred - truth)^2)) AS rmse, count(*) AS n
        FROM rows GROUP BY 1, 2 ORDER BY 1, 2""").df().pivot(index="model", columns="bucket", values=["rmse", "n"])
    order = [f"{a}-{b}h" for a, b in BUCKETS]
    return t.reindex(columns=[(v, b) for v in ("rmse", "n") for b in order if (v, b) in t.columns])


def paired_skill_ci(con, model, ref, n_boot=500, seed=0):
    """Skill = 1 - RMSE_model/RMSE_ref on rows both forecast; CI by resampling issue days."""
    per_day = con.execute("""
        SELECT date_trunc('day', a.issue_time) AS day, sum((a.pred - a.truth)^2) AS se_m, sum((b.pred - a.truth)^2) AS se_r, count(*) AS n
        FROM rows a JOIN rows b ON a.issue_time = b.issue_time AND date_trunc('second', a.time) = date_trunc('second', b.time)
             AND round(a.lat, 3) = round(b.lat, 3) AND round(a.lon, 3) = round(b.lon, 3)  -- model files store float32 coords
             AND a.target_id IS NOT DISTINCT FROM b.target_id  -- RO rows have NULL target_id
        WHERE a.model = ? AND b.model = ? GROUP BY 1""", [model, ref]).fetchnumpy()
    if len(per_day["n"]) == 0:
        return np.nan, (np.nan, np.nan), 0
    se_m, se_r = per_day["se_m"], per_day["se_r"]
    skill = 1 - np.sqrt(se_m.sum() / se_r.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(se_m), size=(n_boot, len(se_m)))
    boots = 1 - np.sqrt(se_m[idx].sum(1) / se_r[idx].sum(1))
    return skill, tuple(np.percentile(boots, [2.5, 97.5])), int(per_day["n"].sum())


def calibration(con):
    """Coverage of ±1σ / ±2σ and mean σ vs RMSE, for models that emit σ."""
    return con.execute("""
        SELECT model, count(*) AS n, round(avg((abs(truth - pred) <= sigma)::INT), 3) AS cov1,
               round(avg((abs(truth - pred) <= 2 * sigma)::INT), 3) AS cov2,
               round(avg(sigma), 3) AS mean_sigma, round(sqrt(avg((truth - pred)^2)), 3) AS rmse
        FROM rows WHERE sigma IS NOT NULL AND sigma > 0 GROUP BY 1 ORDER BY 1""").df()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--mode", default="holdout")
    ap.add_argument("--kind", default="iono")
    ap.add_argument("--ref", default="iri")
    ap.add_argument("--start"); ap.add_argument("--end")
    a = ap.parse_args()
    for var in ("fof2", "hmf2", "mufd"):
        con = connect(a.paths, a.mode, a.kind, var, start=a.start, end=a.end)
        meta = con.execute("SELECT count(DISTINCT issue_time), min(issue_time)::DATE, max(issue_time)::DATE FROM rows").fetchone()
        if var == "fof2":
            print(f"# mode={a.mode} kind={a.kind}  issue times: {meta[0]}  {meta[1]}..{meta[2]}")
        t = rmse_by_bucket(con)
        if t.empty:
            continue
        print(f"\n## {var} RMSE by lead")
        print(t["rmse"].round(3).to_string())
        print("n:", t["n"].iloc[0].astype(int).to_dict())
        print(f"\n## {var} skill vs {a.ref}, all leads, paired on common rows, 95% CI over issue days")
        for m in sorted(t.index):
            if m == a.ref:
                continue
            s, (lo, hi), n = paired_skill_ci(con, m, a.ref)
            print(f"  {m:>16}: {s:+.3f}  [{lo:+.3f}, {hi:+.3f}]  n={n:,}")
        cal = calibration(con)
        if len(cal):
            print(f"\n## {var} σ calibration (ideal cov1 0.683, cov2 0.954; mean σ ≈ RMSE)")
            print(cal.to_string(index=False))
        con.close()


if __name__ == "__main__":
    main()
