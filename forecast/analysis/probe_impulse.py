"""Impulse-response probe: how one station's observation spreads through space and lead time in a checkpoint.

    uv run --extra train analysis/probe_impulse.py run --ckpt /kass/forecast/models/mids_mids_d/best.pt \
        --samples /kass/forecast/samples/val_v9b --out /kass/forecast/eval/probes/impulse_midsd.npz [--issues 40 --stations 6]
    uv run analysis/probe_impulse.py report /kass/forecast/eval/probes/impulse_midsd.npz

For each sampled issue time and fresh station (a token within the last hour), add DELTA to that station's
normalised foF2 anomaly in all of its tokens and decode a 2-degree global grid at LEADS. The response
R = (pred_perturbed - pred_base) / DELTA is the model's effective kernel for that station: 1 at a pixel means
the full input anomaly is carried there. The model has two routes from a station to a query, so every case
runs three ways: 'both' (as in production), 'attn' (tokens perturbed, query state from the unperturbed
inputs) and 'qstate' (only the hand-built nearest-station query features see the perturbation).

report: gain at the station, radial profile, zonal vs meridional reach, and the longitude drift of the
response with lead (0 deg/h = earth-fixed, +15 deg/h = moving with local time, i.e. westward... see below).
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

LEADS = (0, 1, 3, 6, 12, 24)
DELTA = 0.33  # normalised foF2 anomaly units (x1.5 MHz): ~0.5 MHz, inside the training spread
VARIANTS = ("both", "attn", "qstate")
LAT = np.arange(-90, 91, 2.0); LON = np.arange(-180, 180, 2.0)


def gc_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    return 6371 * np.arccos(np.clip(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2), -1, 1))


def tok_stations(tok):
    """Station key (rounded lat, lon in -180..180) for every token."""
    lat = tok[:, 0] * 90.0; lon = np.degrees(np.arctan2(tok[:, 1], tok[:, 2]))
    return np.round(np.c_[lat, lon], 2)


def qstate_for(tok, issue, qlat, qlon):
    from build_samples import query_state
    key = tok_stations(tok)
    times = issue + (tok[:, 5] * 24 * 3600).round().astype("timedelta64[s]")
    return query_state(key[:, 0], key[:, 1], times, tok[:, 9:12].astype(float), issue, qlat, qlon)


def run(a):
    import torch
    from baselines.iri_cache import iri_points_cached
    from build_samples import query_feats
    from predict import load
    from train import pad_tok, subsample
    torch.set_num_threads(a.threads)
    model = load(a.ckpt, "cpu"); args = torch.load(a.ckpt, map_location="cpu", weights_only=False)["args"]
    files = sorted(glob.glob(f"{a.samples}/*.npz"))
    files = files[:: max(1, len(files) // a.issues)][: a.issues]
    glon, glat = np.meshgrid(LON, LAT); glat, glon = glat.ravel(), glon.ravel()
    rng = np.random.default_rng(0)
    out = {k: [] for k in ("resp", "issue", "slat", "slon", "slt", "variant")}

    def decode(tok, glob_, ts, q):
        T = torch.from_numpy(tok)[None]; G = torch.from_numpy(glob_)[None]; S = torch.from_numpy(ts)[None]
        with torch.no_grad():
            H, PAD = model.encode(T, torch.ones(1, len(tok), dtype=torch.bool), G, None, None, S, torch.ones(1, len(ts), dtype=torch.bool))
            return np.concatenate([model.decode(torch.from_numpy(q[i:i + 8192])[None], H, PAD)[0][0, :, 0].numpy() for i in range(0, len(q), 8192)])

    for n, f in enumerate(files):
        z = np.load(f); issue = np.datetime64(int(z["issue"]), "s")
        tok = pad_tok(z["tok"]); ts = z["tok_s"] if model.spots else np.zeros((0, model.f_spot), np.float32)
        glob_ = np.nan_to_num(z["glob"]).astype(np.float32); f107 = float(glob_[0] * 200)
        keys = tok_stations(tok)
        fresh = np.unique(keys[(tok[:, 5] * 24 >= -1.0) & (tok[:, 12] > 0.5)], axis=0)
        if len(fresh) == 0:
            continue
        pick = fresh[rng.choice(len(fresh), min(a.stations, len(fresh)), replace=False)]
        # Fixed per issue: the token subsample (same indices for base and perturbed), queries and IRI per lead.
        keep = np.arange(len(tok)) if len(tok) <= args["max_tok"] else np.sort(np.random.default_rng(int(z["issue"])).choice(len(tok), args["max_tok"], replace=False))
        qs_base = qstate_for(tok, issue, glat, glon)
        qbase = {}
        for L in LEADS:
            tv = issue + np.timedelta64(L * 3600, "s"); times = np.full(len(glat), tv)
            iri = iri_points_cached(times, glat, glon, f107); iri_v = np.c_[iri["fof2"], iri["hmf2"], iri["mufd"]]
            q = query_feats(times, glat, glon, iri_v, np.full(len(glat), 2)); q[:, 5] = L / 24.0
            qbase[L] = q
        base = {L: decode(tok[keep], glob_, ts, np.c_[qbase[L], qs_base].astype(np.float32)) for L in LEADS}
        for s in pick:
            tp = tok.copy(); m = np.all(keys == s, 1) & (tok[:, 12] > 0.5)
            tp[m, 9] += DELTA
            qs_p = qstate_for(tp, issue, glat, glon)
            lt = ((issue - issue.astype("datetime64[D]")).astype(float) / 3600 + s[1] / 15) % 24
            for v in a.variants.split(","):
                t_in = tp if v in ("both", "attn") else tok; qs = qs_p if v in ("both", "qstate") else qs_base
                r = np.stack([(decode(t_in[keep], glob_, ts, np.c_[qbase[L], qs].astype(np.float32)) - base[L]) / DELTA for L in LEADS])
                out["resp"].append(r.astype(np.float16)); out["variant"].append(v)
                out["issue"].append(issue); out["slat"].append(s[0]); out["slon"].append(s[1]); out["slt"].append(lt)
        print(f"{issue}  {len(pick)} stations  ({n + 1}/{len(files)})", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, leads=np.array(LEADS), lat=LAT, lon=LON, delta=DELTA, **{k: np.array(v) for k, v in out.items()})
    print(f"{len(out['resp'])} response fields -> {a.out}")


def report(a):
    z = np.load(a.path); R = z["resp"].astype(np.float32); leads = z["leads"]
    glon, glat = np.meshgrid(z["lon"], z["lat"]); glat, glon = glat.ravel(), glon.ravel()
    variant = z["variant"]; slat, slon = z["slat"], z["slon"]
    edges = np.array([0, 250, 500, 1000, 1500, 2000, 3000, 5000, 20000])
    # Zonal (E-W) and meridional (N-S) reach: pixels within 15 deg of the station's latitude row vs its longitude column.
    print(f"{(variant == 'both').sum()} station impulses, delta {float(z['delta'])} (normalised foF2); R = transferred fraction of the input anomaly\n")
    for v in [v for v in VARIANTS if (variant == v).any()]:
        sel = variant == v
        print(f"=== {v}: mean R by distance from the station (km), rows = lead h")
        print("lead  " + " ".join(f"{a_:>5}-{b:<5}" for a_, b in zip(edges[:-1], edges[1:])))
        for j, L in enumerate(leads):
            prof = np.zeros(len(edges) - 1); cnt = np.zeros(len(edges) - 1)
            for i in np.nonzero(sel)[0]:
                d = gc_km(glat, glon, slat[i], slon[i]); b = np.digitize(d, edges) - 1
                prof += np.bincount(b, R[i, j], len(edges))[: len(edges) - 1]; cnt += np.bincount(b, None, len(edges))[: len(edges) - 1]
            print(f"{L:>4}  " + " ".join(f"{x:>11.3f}" for x in prof / np.maximum(cnt, 1)))
        print()
    print("=== 'both': reach along the latitude row (zonal) vs longitude column (meridional), R by distance, lead 0 / 6 / 24 h")
    sel = np.nonzero(variant == "both")[0]
    for j in [list(leads).index(L) for L in (0, 6, 24)]:
        zon = np.zeros(len(edges) - 1); mer = np.zeros(len(edges) - 1); cz = np.zeros_like(zon); cm = np.zeros_like(zon)
        for i in sel:
            d = gc_km(glat, glon, slat[i], slon[i]); b = np.digitize(d, edges) - 1
            dlon = ((glon - slon[i] + 180) % 360) - 180
            zr = np.abs(glat - slat[i]) <= 4; mr = np.abs(dlon) <= 4 / max(np.cos(np.radians(slat[i])), 0.2)
            zon += np.bincount(b[zr], R[i, j][zr], len(edges))[: len(edges) - 1]; cz += np.bincount(b[zr], None, len(edges))[: len(edges) - 1]
            mer += np.bincount(b[mr], R[i, j][mr], len(edges))[: len(edges) - 1]; cm += np.bincount(b[mr], None, len(edges))[: len(edges) - 1]
        print(f"lead {leads[j]:>2} zonal       " + " ".join(f"{x:>7.3f}" for x in zon / np.maximum(cz, 1)))
        print(f"        meridional  " + " ".join(f"{x:>7.3f}" for x in mer / np.maximum(cm, 1)))
    # Drift: longitude offset of the response's weighted centroid within the station's +-15 deg latitude band,
    # weights R^2 (sign-free), relative to the station. Negative = westward. The sun moves westward over the
    # ground at 15 deg/h, so an anomaly tied to local time shows up 15*lead degrees *west* of the station.
    # Two drift measures in the station's window (+-15 deg lat, +-60 deg lon), longitude east +:
    #  centroid: of the positive response, relative to the station;
    #  shift: the zonal shift s of the lead-0 field that best correlates with the lead-L field (s in -60..60).
    # The sun crosses the ground westward at 15 deg/h, so an anomaly tied to local time moves -15 deg/h.
    print("\n=== drift with lead, window +-15 deg lat x +-60 deg lon around the station (deg of longitude, east +)")
    print("variant lead  centroid(median)  shift(median)  shift deg/h   (earth-fixed 0, local-time-fixed -15)")
    nlat, nlon = len(z["lat"]), len(z["lon"])
    shifts = np.arange(-30, 31)  # grid columns (2 deg each)
    for v in [v for v in VARIANTS if (variant == v).any()]:
        idx = np.nonzero(variant == v)[0]
        for j, L in enumerate(leads):
            cen, sh = [], []
            for i in idx:
                r0 = R[i, 0].reshape(nlat, nlon); rl = R[i, j].reshape(nlat, nlon)
                ci = int(np.argmin(np.abs(z["lon"] - ((slon[i] + 180) % 360 - 180)))); rows = np.abs(z["lat"] - slat[i]) <= 15
                cols = (ci + np.arange(-30, 31)) % nlon
                wl = np.clip(rl[np.ix_(rows, cols)], 0, None)
                if wl.sum() > 0:
                    cen.append((wl.sum(0) * np.arange(-30, 31) * 2).sum() / wl.sum())
                a0 = r0[np.ix_(rows, cols)].ravel()
                cc = [np.corrcoef(np.roll(r0, s, 1)[np.ix_(rows, cols)].ravel(), rl[np.ix_(rows, cols)].ravel())[0, 1] if a0.std() > 0 else np.nan for s in shifts]
                if np.isfinite(cc).any():
                    sh.append(shifts[int(np.nanargmax(cc))] * 2)
            c, s_ = np.median(cen) if cen else np.nan, np.median(sh) if sh else np.nan
            print(f"{v:>7} {L:>4}  {c:>16.1f}  {s_:>13.1f}  {(s_ / L if L else 0):>11.2f}")
    print("\n=== gain at the station's own pixel (nearest 2-deg cell), 'both', by lead: mean, and by station local time at issue")
    lt = z["slt"]
    for j, L in enumerate(leads):
        g = np.array([R[i, j][np.argmin(gc_km(glat, glon, slat[i], slon[i]))] for i in sel])
        by = {f"LT{a_:02d}-{a_ + 6:02d}": g[(lt[sel] >= a_) & (lt[sel] < a_ + 6)].mean() for a_ in (0, 6, 12, 18)}
        print(f"lead {L:>2}: {g.mean():.3f}   " + "  ".join(f"{k} {x:.3f}" for k, x in by.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--ckpt", required=True); r.add_argument("--samples", required=True); r.add_argument("--out", required=True)
    r.add_argument("--issues", type=int, default=40); r.add_argument("--stations", type=int, default=6); r.add_argument("--threads", type=int, default=8)
    r.add_argument("--variants", default=",".join(VARIANTS), help="subset of both,attn,qstate (a no-qstate model only needs both)")
    p = sp.add_parser("report"); p.add_argument("path")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else report(a)
