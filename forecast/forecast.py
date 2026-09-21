"""Eyeball tool (PLAN.md Phase 1): render a 24-h forecast map set for any model at a historical issue time.

    uv run forecast.py --at 2025-06-15T12:00 --model anomaly_decay --out /kass/forecast/maps/2025-06-15
    uv run --extra train forecast.py --at 2025-06-15T12:00 --model /kass/forecast/models/v0/best.pt --out ...

Per lead (0..24 h, --step-hours): three panels for --var (default fof2): forecast map, anomaly vs IRI,
uncertainty (σ, checkpoint models only), with input observations as dots on the anomaly panel (coloured
by their own anomaly) and the observations that actually happened at that lead as dots on the forecast
panel (coloured by value, same scale). Writes PNGs and an animated GIF. `--at now` pulls the last 24 h of ionosonde data from the prop.kc2g.com API instead of the snapshot.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines.iri_cache import iri_points_cached  # noqa: E402
from baselines.simple import MODELS, VARS  # noqa: E402
from data.drivers import attach_essn, attach_indices, f107_trailing  # noqa: E402
from data.load import SNAPSHOT, connect, inputs_at, targets_after  # noqa: E402
from train.build_samples import ANOM_SCALE, query_feats, tokens_at  # noqa: E402

LAT = np.arange(-90, 91, 1.0)
LON = np.arange(-180, 181, 1.0)
LIMITS = {"fof2": (1, 16), "hmf2": (180, 450), "mufd": (5, 45)}
ANOM_LIM = {"fof2": 3.0, "hmf2": 60.0, "mufd": 8.0}


API = "https://prop.kc2g.com/api"


def live_inputs(con, t):
    """Inputs for issue time now, from the production API: every station with data in the last 24 h."""
    import json
    import urllib.request
    from data.load import BOUNDS
    st = con.execute("SELECT id, lat, lon, cluster FROM station").df().set_index("id")
    since = (pd.Timestamp(t) - pd.Timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with urllib.request.urlopen(f"{API}/stations.json?maxage=86400", timeout=60) as r:
        active = {(m["station"]["id"] if isinstance(m["station"], dict) else m["station"]) for m in json.load(r)}
    rows = []
    for sid in sorted(active):
        if sid not in st.index:
            continue
        with urllib.request.urlopen(f"{API}/sonde_export?station={sid}&since={since}&format=json", timeout=120) as r:
            for m in json.load(r):
                rows.append({"station_id": sid, "cluster": st.cluster[sid], "time": pd.Timestamp(m["time"]), "lat": st.lat[sid], "lon": st.lon[sid],
                             "cs": -1 if m["cs"] is None else m["cs"], **{v: m[v] for v in VARS}})
    df = pd.DataFrame(rows)
    df = df[(df.cs >= 75) | (df.cs == -1)]
    for v, (lo, hi) in BOUNDS.items():
        if v in df:
            df[v] = df[v].where((df[v] >= lo) & (df[v] <= hi))
    df = df[df["time"] <= pd.Timestamp(t) - pd.Timedelta(minutes=15)]
    print(f"live: {len(df)} rows from {df.station_id.nunique()} stations since {since}")
    return df


def grid_queries(t_valid, f107):
    lon, lat = np.meshgrid(LON, LAT)
    lat, lon = lat.ravel(), lon.ravel()
    times = np.full(len(lat), np.datetime64(t_valid, "s"))
    iri = iri_points_cached(times, lat, lon, f107)
    iri_v = np.c_[iri["fof2"], iri["hmf2"], iri["mufd"]]
    return lat, lon, times, iri_v


def run_checkpoint(ckpt, tok, glob, lat, lon, times, iri_v, lead_h, dev, qstate=None):
    import torch
    from train.model import AnomalyModel
    ck = torch.load(ckpt, map_location=dev)
    model = AnomalyModel(d=ck["args"]["d"], enc_layers=ck["args"]["layers"], query_self_attn=ck["args"].get("query_self_attn", True), glotec=ck["args"].get("glotec", False), spots=ck["args"].get("spots", False), f_spot=ck["args"].get("f_spot", 29), f_qry=ck["args"].get("f_qry", 10)).to(dev); model.load_state_dict(ck["model"]); model.eval()
    q = query_feats(times, lat, lon, iri_v, np.full(len(lat), 2)); q[:, 5] = lead_h / 24.0
    if model.f_qry > q.shape[1]:
        q = np.c_[q, qstate].astype(np.float32)
    T = torch.from_numpy(tok)[None].to(dev); M = torch.ones(1, len(tok), dtype=torch.bool, device=dev); G = torch.from_numpy(glob)[None].to(dev)
    means, sig = [], []
    with torch.no_grad():
        H, PAD = model.encode(T, M, G)
        for i in range(0, len(q), 8192):
            m, lv = model.decode(torch.from_numpy(q[i:i + 8192])[None].to(dev), H, PAD)
            means.append(m[0].cpu().numpy()); sig.append(np.exp(0.5 * lv[0].cpu().numpy()))
    return np.concatenate(means) * ANOM_SCALE, np.concatenate(sig) * ANOM_SCALE


def render(out, t_issue, lead_h, var, fc, iri, sigma, inputs, truth, model_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vi = VARS.index(var); lo, hi = LIMITS[var]; al = ANOM_LIM[var]
    ncol = 3 if sigma is not None else 2
    fig, axes = plt.subplots(1, ncol, figsize=(6.2 * ncol, 4.2), constrained_layout=True)
    ext = [-180, 180, -90, 90]
    ax = axes[0]; im = ax.imshow(fc[:, vi].reshape(181, 361), origin="lower", extent=ext, vmin=lo, vmax=hi, cmap="viridis", aspect="auto")
    if truth is not None and len(truth):
        ax.scatter(truth.lon, truth.lat, c=truth[var], vmin=lo, vmax=hi, cmap="viridis", s=28, edgecolors="white", linewidths=0.6)
    ax.set_title(f"{model_name}: {var} at T+{lead_h:g} h  (dots: observations at that hour)"); fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axes[1]; im = ax.imshow((fc[:, vi] - iri[:, vi]).reshape(181, 361), origin="lower", extent=ext, vmin=-al, vmax=al, cmap="RdBu_r", aspect="auto")
    if inputs is not None and len(inputs):
        ax.scatter(inputs.lon, inputs.lat, c=inputs.anom_fof2 if var == "fof2" else 0, vmin=-al, vmax=al, cmap="RdBu_r", s=22, edgecolors="black", linewidths=0.5)
    ax.set_title(f"anomaly vs IRI  (dots: input stations, their own anomaly at T)"); fig.colorbar(im, ax=ax, shrink=0.8)
    if sigma is not None:
        ax = axes[2]; im = ax.imshow(sigma[:, vi].reshape(181, 361), origin="lower", extent=ext, vmin=0, vmax=al, cmap="magma", aspect="auto")
        ax.set_title("σ"); fig.colorbar(im, ax=ax, shrink=0.8)
    for ax in axes:
        ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-90, 91, 30)); ax.grid(alpha=0.25)
    fig.suptitle(f"issue {t_issue}   lead {lead_h:g} h   valid {pd.Timestamp(t_issue) + pd.Timedelta(hours=lead_h)}")
    f = out / f"{var}_lead{int(lead_h):02d}.png"; fig.savefig(f, dpi=110); plt.close(fig)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", required=True, help="issue time, e.g. 2025-06-15T12:00 (from the snapshot), or 'now' (live prop.kc2g.com API)")
    ap.add_argument("--model", default="anomaly_decay", help="a baseline name from baselines.simple.MODELS, or a checkpoint .pt")
    ap.add_argument("--var", default="fof2", choices=VARS)
    ap.add_argument("--step-hours", type=float, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-inputs", action="store_true", help="drop every observation token: the model's learned climatology (indices + geometry) alone")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    con = connect(); attach_indices(con, SNAPSHOT); attach_essn(con, SNAPSHOT)
    live = a.at == "now"
    t = str(pd.Timestamp.utcnow().floor("min").tz_localize(None)) if live else str(pd.Timestamp(a.at))
    f107 = f107_trailing(con, t)
    rng = np.random.default_rng(0)
    pool = a.model.endswith(".pt") and bool(__import__("torch").load(a.model, map_location="cpu", weights_only=False)["args"].get("pool", False))
    tok, _, inputs = tokens_at(con, t, f107, rng, holdout_frac=0.0, max_tok=4096, inp=live_inputs(con, t) if live else None, pool=pool)
    if a.no_inputs:
        tok = tok[:0]; inputs = inputs.iloc[:0]
    recent = inputs[inputs["time"].to_numpy() >= np.datetime64(t, "s") - np.timedelta64(2, "h")].groupby("cluster").agg(lat=("lat", "first"), lon=("lon", "first"), anom_fof2=("anom_fof2", "mean")).reset_index()
    targets = targets_after(con, t).df() if not live else pd.DataFrame(columns=["kind", "time", "lat", "lon", *VARS])
    for df in (recent, targets):  # station longitudes are stored 0..360; the map frame is -180..180
        df["lon"] = ((df["lon"] + 180) % 360) - 180
    is_ckpt = a.model.endswith(".pt")
    if is_ckpt:
        # global token, built the same way as build_samples (indices known at T)
        daily = con.execute("SELECT date, f107_obs FROM daily").df().set_index("date")
        ap8 = con.execute("SELECT date, ap3h FROM daily ORDER BY date").df()
        ap_series = pd.Series(np.concatenate(ap8.ap3h.to_list()).astype(float), index=np.repeat(pd.to_datetime(ap8.date), 8) + pd.to_timedelta(np.tile(np.arange(8) * 3, len(ap8)), unit="h"))
        ts = pd.Timestamp(t); slot = ts.floor("3h"); ap_now = float(ap_series.get(slot, np.nan)); ap_max = float(ap_series.loc[slot - pd.Timedelta(hours=21):slot].max())
        f107d = float(daily.f107_obs.get(pd.Timestamp(ts.date()) - pd.Timedelta(days=1), np.nan)); f107d = f107 if not np.isfinite(f107d) else f107d
        doy = ts.dayofyear / 365.25 * 2 * np.pi
        glob = np.array([f107 / 200.0, f107d / 200.0, np.log1p(np.nan_to_num(ap_now, nan=ap_max)) / 5, np.log1p(ap_max) / 5, np.sin(doy), np.cos(doy)], np.float32)
        name = Path(a.model).parent.name + (" (indices only)" if a.no_inputs else "")
    else:
        from types import SimpleNamespace
        name = a.model
    frames = []
    for lead in np.arange(0, 24.001, a.step_hours):
        t_valid = pd.Timestamp(t) + pd.Timedelta(hours=float(lead))
        lat, lon, times, iri_v = grid_queries(t_valid, f107)
        sigma = None
        if is_ckpt:
            from train.build_samples import query_state
            qs = query_state(inputs.lat.to_numpy(), inputs.lon.to_numpy(), inputs["time"].to_numpy().astype("datetime64[s]"), np.stack(inputs.anom_n.to_list()) if len(inputs) else np.zeros((0, 3)), np.datetime64(t, "s"), lat, lon) if is_ckpt else None
            anom, sigma = run_checkpoint(a.model, tok, glob, lat, lon, times, iri_v, float(lead), a.device, qstate=qs)
            fc = iri_v + anom
        else:
            qdf = pd.DataFrame({"kind": "grid", "target_id": np.nan, "cluster": np.nan, "time": times, "lat": lat, "lon": lon})
            ctx = SimpleNamespace(con=con, t=t, inputs=inputs, targets=qdf, iri={"fof2": iri_v[:, 0], "hmf2": iri_v[:, 1], "mufd": iri_v[:, 2]}, f107=f107)
            r = MODELS[a.model](ctx); fc = np.c_[r["fof2"], r["hmf2"], r["mufd"]]
        w = np.abs((targets["time"].to_numpy().astype("datetime64[s]") - np.datetime64(t_valid, "s")).astype(float)) <= 1800
        truth = targets[w & (targets["kind"] == "iono")]
        np.savez_compressed(out / f"grid_lead{int(lead):02d}.npz", fc=fc.astype(np.float32), iri=iri_v.astype(np.float32),
                            sigma=(sigma.astype(np.float32) if sigma is not None else np.zeros(0, np.float32)), lead_h=float(lead),
                            issue=np.datetime64(t, "s").astype(np.int64), stations=recent[["lat", "lon"]].to_numpy(np.float32))
        frames.append(render(out, t, float(lead), a.var, fc, iri_v, sigma, recent, truth, name))
        print(f"lead {lead:g} h: {a.var} range {fc[:, VARS.index(a.var)].min():.1f}..{fc[:, VARS.index(a.var)].max():.1f}", flush=True)
    from PIL import Image
    imgs = [Image.open(f) for f in frames]
    imgs[0].save(out / f"{a.var}.gif", save_all=True, append_images=imgs[1:], duration=700, loop=0)
    print(f"{len(frames)} frames -> {out}/{a.var}.gif")


if __name__ == "__main__":
    main()
