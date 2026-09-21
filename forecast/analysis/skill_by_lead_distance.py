"""foF2 RMSE vs lead and vs distance to the nearest contributing ionosonde, for a model, the production GP,
the tuned kernel and IRI, on a window where all are out of sample.

    uv run analysis/skill_by_lead_distance.py --model /kass/forecast/model_v4.parquet --name v4 --start 2025-10-01 --end 2026-01-01 --out /kass/forecast/eval/skill_v4.json

"Contributing" = a station with data in the 24 h input window at that issue time and not withheld, i.e. the
model's token set; recovered from the sample files' token geometry. GP rows at its own assimilated stations
(full mode) are distance 0 by definition; the distance chart therefore uses RO points (fair for all) and
holdout ionosonde rows (model / kernel / IRI).
"""
import argparse
import glob
import json
import os

import duckdb
import numpy as np
import pandas as pd

DIST_EDGES = [0, 250, 500, 1000, 2000, 4000, 20000]
DIST_LABELS = ["<250", "250–500", "500–1000", "1000–2000", "2000–4000", ">4000"]


def gc_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    return 6371 * np.arccos(np.clip(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2), -1, 1))


def station_sets(samples, start, end):
    out = {}
    for f in sorted(glob.glob(f"{samples}/*.npz")):
        b = os.path.basename(f)[:13]
        if not (start.replace("-", "") <= b[:8] < end.replace("-", "")):
            continue
        z = np.load(f); tok = z["tok"]
        lat = tok[:, 0] * 90; lon = np.degrees(np.arctan2(tok[:, 1], tok[:, 2]))
        pts = np.unique(np.round(np.c_[lat, lon], 2), axis=0)
        out[pd.Timestamp(b[:4] + "-" + b[4:6] + "-" + b[6:8] + " " + b[9:11] + ":" + b[11:13])] = pts
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--samples", default="/kass/forecast/samples/eval2025_g")
    a = ap.parse_args()
    files = [a.model] + glob.glob("/kass/forecast/eval/val2025/base_q*.parquet") + glob.glob("/kass/forecast/eval/val2025/gp_q*.parquet")
    con = duckdb.connect(); con.execute("SET memory_limit='32GB'; SET TimeZone='UTC'")
    df = con.execute(f"""SELECT issue_time, mode, model, kind, time, lead_h, lat, lon, truth, pred
        FROM read_parquet({[str(f) for f in files]}, union_by_name=true)
        WHERE var='fof2' AND lead_h <= 24 AND issue_time >= '{a.start}' AND issue_time < '{a.end}'
          AND model IN ('{a.name}', 'gp_prod', 'anomaly_decay', 'iri') AND NOT isnan(pred)""").df()
    df["model"] = df["model"].replace({a.name: "model"})
    sets = station_sets(a.samples, a.start, a.end)
    dist = np.full(len(df), np.nan)
    for t, idx in df.groupby("issue_time").indices.items():
        pts = sets.get(pd.Timestamp(t))
        if pts is None:
            continue
        la, lo = df.lat.to_numpy()[idx], ((df.lon.to_numpy()[idx] + 180) % 360) - 180
        dist[idx] = np.min(gc_km(la[:, None], lo[:, None], pts[None, :, 0], pts[None, :, 1]), axis=1)
    df["dist"] = dist
    df["se"] = (df.pred - df.truth) ** 2
    res = {"window": [a.start, a.end], "model_name": a.name}
    # by lead (hourly bins), two views
    for view, sel in (("iono_full", (df["mode"] == "full") & (df.kind == "iono")), ("ro", df.kind == "ro"), ("iono_holdout", (df["mode"] == "holdout") & (df.kind == "iono"))):
        d = df[sel].copy(); d["lead"] = np.ceil(d.lead_h).clip(1, 24).astype(int)
        g = d.groupby(["model", "lead"]).se.mean().pow(0.5).unstack("lead")
        res[f"lead_{view}"] = {m: [round(float(g.loc[m, h]), 4) if h in g.columns and np.isfinite(g.loc[m, h]) else None for h in range(1, 25)] for m in g.index}
        res[f"lead_{view}_n"] = int(len(d) / max(d.model.nunique(), 1))
    # by distance, two views
    for view, sel in (("ro", df.kind == "ro"), ("iono_holdout", (df["mode"] == "holdout") & (df.kind == "iono"))):
        d = df[sel & np.isfinite(df.dist)].copy(); d["bin"] = pd.cut(d.dist, DIST_EDGES, labels=DIST_LABELS, right=False)
        g = d.groupby(["model", "bin"], observed=True).se.mean().pow(0.5).unstack("bin")
        n = d[d.model == d.model.iloc[0]].groupby("bin", observed=True).size()
        res[f"dist_{view}"] = {m: [round(float(g.loc[m, b]), 4) if b in g.columns and np.isfinite(g.loc[m, b]) else None for b in DIST_LABELS] for m in g.index}
        res[f"dist_{view}_n"] = [int(n.get(b, 0)) for b in DIST_LABELS]
    json.dump(res, open(a.out, "w"), indent=1)
    for k, v in res.items():
        if isinstance(v, dict):
            print(k); [print(f"  {m:>14}", [x if x is None else round(x, 2) for x in vals]) for m, vals in v.items()]
        elif isinstance(v, list) and k.endswith("_n"):
            print(k, v)


if __name__ == "__main__":
    main()
