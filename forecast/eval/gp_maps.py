"""Score archived map runs (production GP 'assimilated' + 'irimap', or an experiment's runs) at the harness targets -> replay schema.

    uv run eval/gp_maps.py --start 2025-01-01 --end 2025-04-01 --step-hours 3 --out /kass/forecast/eval/val2025/gp_q1.parquet

For issue time T: the production run with target_time == T (experiment IS NULL), its 25 hourly map files
(valid times T .. T+24h) under /kass/prop-archive/prop-archive/<id//1e6>/<id//1e3 % 1e3>/<id>/{assimilated,irimap}/.
Targets are targets_after(T) exactly as replay.py uses; maps are bilinear in space and linear in time.
mode: 'full' if the target station was assimilated by that run (listed in /stationdata/pred), else 'holdout'
(a station production did not use, so its map value there is a genuine spatial prediction); RO rows are 'full'
like the harness. sigma = GP stdev map (assimilated only).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.load import SNAPSHOT, connect, targets_after  # noqa: E402

ARCHIVE = Path(os.environ.get("PROP_ARCHIVE", "/kass/prop-archive/prop-archive"))  # offsite tree; on the server /archive/<id> is the flat pre-upload layout
VARS = ("fof2", "hmf2", "mufd")


def run_dir(run_id):
    flat = ARCHIVE / str(run_id)  # server-side /archive/<id> (runs archived but not yet uploaded)
    return flat if flat.exists() else ARCHIVE / str(run_id // 10**6) / f"{(run_id // 1000) % 1000:03d}" / str(run_id)


def bilinear(m, lat, lon):
    """m (181, 361) on lat -90..90, lon -180..180 (inclusive, col 360 == col 0)."""
    y = np.clip(lat + 90, 0, 180); x = np.mod(lon + 180, 360)
    y0 = np.minimum(y.astype(int), 179); x0 = np.minimum(x.astype(int), 359); fy = y - y0; fx = x - x0
    return ((1 - fy) * (1 - fx) * m[y0, x0] + (1 - fy) * fx * m[y0, x0 + 1] + fy * (1 - fx) * m[y0 + 1, x0] + fy * fx * m[y0 + 1, x0 + 1])


def load_maps(d):
    """{valid_ts: {var: map}} for one product dir; plus stdev fof2 and the assimilated station coords if present."""
    out, sd, stations = {}, {}, None
    for f in sorted(d.glob("*.h5")):
        with h5py.File(f) as h:
            ts = int(round(float(h["ts"][()])))
            out[ts] = {v: h[f"/maps/{v}"][:] for v in VARS}
            if "/stdev/fof2" in h:
                sd[ts] = {"fof2": h["/stdev/fof2"][:], "hmf2": h["/stdev/hmf2"][:]}
            if stations is None and "/stationdata/pred" in h:
                stations = [(round(s["station.latitude"], 2), round(s["station.longitude"], 2)) for s in json.loads(h["/stationdata/pred"][()])]
    return out, sd, stations


def score_issue(con, runs, t, name_map):
    T = pd.Timestamp(t)
    r = runs[runs.target_time == T]
    if r.empty:
        return []
    run_id = int(r.id.min())
    tg = targets_after(con, str(t)).df()
    if tg.empty:
        return []
    qt = tg["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    lat, lon = tg.lat.to_numpy(), tg.lon.to_numpy()
    rows = []
    assimilated = None  # station set comes from the assimilated product; irimap files carry none
    for product, model_name in name_map.items():
        d = run_dir(run_id) / product
        if not d.exists():
            continue
        maps, sd, stations = load_maps(d)
        ts = np.array(sorted(maps))
        if len(ts) < 2:
            continue
        i1 = np.clip(np.searchsorted(ts, qt), 1, len(ts) - 1); i0 = i1 - 1
        w = np.clip((qt - ts[i0]) / np.maximum(ts[i1] - ts[i0], 1), 0, 1)
        pred = {}; sig = {}
        for v in VARS:
            a = np.zeros(len(qt)); b = np.zeros(len(qt))
            for k in np.unique(i0):
                sel = i0 == k
                a[sel] = bilinear(maps[ts[k]][v], lat[sel], lon[sel]); b[sel] = bilinear(maps[ts[k + 1]][v], lat[sel], lon[sel])
            pred[v] = (1 - w) * a + w * b
            if sd and v in ("fof2", "hmf2"):
                s = np.zeros(len(qt))
                for k in np.unique(i0):
                    sel = i0 == k; s[sel] = 0.5 * (bilinear(sd[ts[k]][v], lat[sel], lon[sel]) + bilinear(sd[ts[k + 1]][v], lat[sel], lon[sel]))
                sig[v] = s * pred[v]  # the GP works in log space: /stdev is σ of log(value); linear σ ≈ value × σ_log
        # station assimilated by this run?  (coords rounded to 0.01°)
        if stations is not None:
            assimilated = set(stations)
        if assimilated is None:
            assimilated = set()
        st_key = list(zip(np.round(lat, 2), np.round(lon, 2)))
        is_in = np.array([k in assimilated for k in st_key])
        kind = tg["kind"].to_numpy()
        mode = np.where((kind == "iono") & ~is_in, "holdout", "full")
        lead = (qt - int(T.timestamp())) / 3600.0
        for v in VARS:
            truth = tg[v].to_numpy(dtype=float); ok = np.isfinite(truth) & np.isfinite(pred[v])
            rows.append(pd.DataFrame({
                "issue_time": np.datetime64(T, "s"), "mode": mode[ok], "model": model_name, "kind": kind[ok],
                "target_id": tg["target_id"].to_numpy(dtype=float)[ok], "cluster": tg["cluster"].to_numpy(dtype=float)[ok],
                "time": tg["time"].to_numpy()[ok], "lead_h": lead[ok], "lat": lat[ok], "lon": lon[ok], "var": v,
                "truth": truth[ok], "pred": pred[v][ok], "sigma": (sig[v][ok] if v in sig else np.nan)}))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True); ap.add_argument("--step-hours", type=float, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--experiment", help="score this experiment's runs (runs.experiment) instead of production; model names get the experiment as suffix")
    ap.add_argument("--runs", default=f"{SNAPSHOT}/runs.parquet", help="runs table as parquet (re-dump for recent experiment runs)")
    a = ap.parse_args()
    con = connect()
    where = "experiment IS NULL" if not a.experiment else f"experiment = '{a.experiment}'"
    runs = con.execute(f"SELECT id, target_time FROM '{a.runs}' WHERE {where} AND target_time >= '{a.start}' AND target_time < '{a.end}'").df()
    names = {"assimilated": "gp_prod", "irimap": "irimap_prod"} if not a.experiment else {"assimilated": f"live_{a.experiment}"}
    frames = []
    times = pd.date_range(a.start, a.end, freq=f"{int(a.step_hours * 60)}min", inclusive="left")
    for i, t in enumerate(times):
        frames += score_issue(con, runs, t, names)
        if i % 50 == 0:
            print(f"{t} ({i + 1}/{len(times)})", flush=True)
    df = pd.concat(frames, ignore_index=True); df.to_parquet(a.out)
    print(f"{len(df):,} rows -> {a.out}")


if __name__ == "__main__":
    main()
