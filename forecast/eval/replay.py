"""Point-in-time replay: run models at historical issue times, score against later observations.

    uv run eval/replay.py --start 2024-01-01 --end 2024-01-08 --step-hours 6 --out /kass/forecast/eval/smoke.parquet

Output rows: issue_time, mode, model, kind, target_id, cluster, time, lead_h, lat, lon, var, truth, pred.
mode = 'holdout' (20% of clusters withheld from inputs, scored on them + RO) or 'full' (all stations in,
scored on RO only for ionosondes... no: scored on everything, since ionosonde targets are future obs).
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.iri_cache import iri_points_cached as iri_points  # noqa: E402
from baselines.simple import MODELS, VARS  # noqa: E402
from data.drivers import attach_essn, attach_indices, f107_trailing  # noqa: E402
from data.load import SNAPSHOT, connect, inputs_at, targets_after  # noqa: E402

HOLDOUT_FRAC = 0.2


def run_issue(con, t, models, seed, held_stations=None):
    inputs = inputs_at(con, t).df()
    targets = targets_after(con, t).df()
    if len(targets) == 0:
        return []
    f107 = f107_trailing(con, t)
    iri_t = iri_points(targets["time"].to_numpy(), targets["lat"].to_numpy(), targets["lon"].to_numpy(), f107)
    rng = np.random.default_rng(seed)
    clusters = np.unique(inputs["cluster"])
    if held_stations is not None:  # reproduce an external holdout set (e.g. production's) by cluster
        held = set(inputs.loc[inputs["station_id"].isin(held_stations), "cluster"])
    else:
        held = set(rng.choice(clusters, int(round(HOLDOUT_FRAC * len(clusters))), replace=False)) if len(clusters) else set()
    lead_h = (targets["time"].to_numpy().astype("datetime64[s]") - np.datetime64(t, "s")).astype(float) / 3600.0
    rows = []
    for mode in ("full", "holdout"):
        inp = inputs[~inputs["cluster"].isin(held)] if mode == "holdout" else inputs
        # In holdout mode, ionosonde targets are only the withheld clusters (RO always scored).
        sel = (targets["kind"] == "ro") | targets["cluster"].isin(held) if mode == "holdout" else np.ones(len(targets), bool)
        sel = np.asarray(sel)
        ctx = SimpleNamespace(con=con, t=t, inputs=inp, targets=targets, iri=iri_t, f107=f107)
        for name, model in models.items():
            pred = model(ctx)
            for v in VARS:
                truth = targets[v].to_numpy()
                ok = sel & np.isfinite(truth)
                if not ok.any():
                    continue
                rows.append(pd.DataFrame({
                    "issue_time": np.datetime64(t, "s"), "mode": mode, "model": name,
                    "kind": targets["kind"].to_numpy()[ok], "target_id": targets["target_id"].to_numpy()[ok],
                    "cluster": targets["cluster"].to_numpy()[ok], "time": targets["time"].to_numpy()[ok],
                    "lead_h": lead_h[ok], "lat": targets["lat"].to_numpy()[ok], "lon": targets["lon"].to_numpy()[ok],
                    "var": v, "truth": truth[ok], "pred": pred[v][ok],
                }))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--step-hours", type=float, default=6)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--snapshot", default=str(SNAPSHOT))
    ap.add_argument("--holdouts", help="parquet with issue_time, station_id: use these issue times and holdout stations")
    a = ap.parse_args()
    con = connect(Path(a.snapshot))
    attach_indices(con, a.snapshot)
    attach_essn(con, a.snapshot)
    models = {m: MODELS[m] for m in a.models.split(",")}
    held_by_t = None
    if a.holdouts:
        h = pd.read_parquet(a.holdouts)
        h = h[(h.issue_time >= a.start) & (h.issue_time < a.end)]
        held_by_t = h.groupby("issue_time")["station_id"].apply(set).to_dict()
        issue_times = sorted(held_by_t)
    else:
        issue_times = pd.date_range(a.start, a.end, freq=f"{int(a.step_hours * 60)}min", inclusive="left")
    frames = []
    for i, t in enumerate(issue_times):
        frames += run_issue(con, str(t), models, seed=int(pd.Timestamp(t).value // 10**9) % 2**32,
                            held_stations=None if held_by_t is None else held_by_t[t])
        print(f"{t}  ({i + 1}/{len(issue_times)})", flush=True)
    df = pd.concat(frames, ignore_index=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out)
    print(f"{len(df):,} rows -> {a.out}")


if __name__ == "__main__":
    main()
