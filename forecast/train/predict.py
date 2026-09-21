"""Run a checkpoint over a sample directory and write replay-schema rows, so eval/report.py and eval/rank.py
score the model exactly like the baselines.

    uv run --extra train train/predict.py --ckpt /kass/forecast/models/v0/best.pt --samples /kass/forecast/samples/eval2025 --out /kass/forecast/eval/model_v0.parquet [--name model_v0]

mode: 'holdout' for queries whose cluster was withheld from the inputs, 'full' otherwise (and for RO).
pred = IRI + predicted anomaly; sigma column carries the predicted std (in physical units).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

from model import AnomalyModel, F_QRY

ANOM_SCALE = np.array([1.5, 40.0, 4.5], np.float32)
VARS = ("fof2", "hmf2", "mufd")


def load(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev)
    model = AnomalyModel(d=ck["args"]["d"], enc_layers=ck["args"]["layers"], query_self_attn=ck["args"].get("query_self_attn", True), glotec=ck["args"].get("glotec", False), spots=ck["args"].get("spots", False), f_spot=ck["args"].get("f_spot", 29), f_qry=ck["args"].get("f_qry", 10)).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    return model


def predict_dir(model, samples, name, dev, bs_q=4096, drop_iono=False, drop_glotec=False, start=None, end=None, drop_spots=False, max_tok=None, recency_tau=None, spot_within=None):
    """drop_iono / drop_glotec: remove that source at prediction time (outage simulation).
    max_tok / recency_tau: the token budget the checkpoint was trained with (train.py --max-tok), applied deterministically."""
    from train import subsample, pad_tok
    frames = []
    with torch.no_grad():
        for f in sorted(glob.glob(f"{samples}/*.npz")):
            b = os.path.basename(f)[:8]
            if (start and b < start) or (end and b >= end):
                continue
            z = np.load(f)
            t_np = pad_tok(z["tok"][:0] if drop_iono else z["tok"])
            if max_tok:
                t_np = subsample(t_np, max_tok, np.random.default_rng(int(z["issue"])), recency_tau)
            tok = torch.from_numpy(t_np)[None].to(dev); tmask = torch.ones(1, len(t_np), dtype=torch.bool, device=dev)
            g = torch.from_numpy(np.nan_to_num(z["glob"]))[None].to(dev)
            tg = z["tok_g"] if ("tok_g" in z.files and not drop_glotec) else np.zeros((0, 14), np.float32)
            tg = tg[tg[:, 7] < 0.5]  # no qf=0 GloTEC cells (as in training since 2026-09-09; older checkpoints saw them as a 25% sample)
            tok_g = torch.from_numpy(tg)[None].to(dev); g_mask = torch.ones(1, len(tg), dtype=torch.bool, device=dev)
            tsp = z["tok_s"] if ("tok_s" in z.files and not drop_spots) else np.zeros((0, model.f_spot), np.float32)
            if spot_within is not None and len(tsp):  # standalone-receiver study: keep only spot cells within R km of (lat, lon)
                clat, clon, rkm = spot_within; la = tsp[:, 0] * 90; lo = np.degrees(np.arctan2(tsp[:, 1], tsp[:, 2]))
                d = 6371 * np.arccos(np.clip(np.sin(np.radians(la)) * np.sin(np.radians(clat)) + np.cos(np.radians(la)) * np.cos(np.radians(clat)) * np.cos(np.radians(lo - clon)), -1, 1))
                tsp = tsp[d <= rkm]
            tok_s = torch.from_numpy(tsp)[None].to(dev); s_mask = torch.ones(1, len(tsp), dtype=torch.bool, device=dev)
            means, lvs = [], []
            H, PAD = model.encode(tok, tmask, g, tok_g, g_mask, tok_s, s_mask)
            qarr = z["qry"]
            if drop_iono and qarr.shape[1] > F_QRY:  # nearest-station state columns (--qstate) are ionosonde data; dist_nearest
                qarr = qarr.copy(); qarr[:, F_QRY:] = 0; qarr[:, F_QRY + 1] = 1.0  # is query_state()'s "none in reach" = 1, not 0
            if os.environ.get("RO_AS_MAP"):  # diagnostic: score RO points with the query kind flag cleared (as a map pixel would be)
                qarr = qarr.copy(); qarr[:, 9] = 0.0
            for i in range(0, len(qarr), bs_q):
                q = torch.from_numpy(qarr[i:i + bs_q])[None].to(dev)
                m, lv = model.decode(q, H, PAD); means.append(m[0].cpu().numpy()); lvs.append(lv[0].cpu().numpy())
            mean = np.concatenate(means) * ANOM_SCALE; sigma = np.exp(0.5 * np.concatenate(lvs)) * ANOM_SCALE
            pred = z["iri_q"] + mean; truth = z["iri_q"] + z["tgt"] * ANOM_SCALE
            issue = np.datetime64(int(z["issue"]), "s"); qtime = z["qtime"].astype("datetime64[s]")
            kind = np.where(z["qkind"] == 1, "ro", "iono"); mode = np.where(z["held"] & (z["qkind"] == 0), "holdout", "full")
            base = dict(issue_time=issue, mode=mode, model=name, kind=kind, target_id=np.where(z["qid"] < 0, np.nan, z["qid"]).astype(float),
                        cluster=np.where(z["qcluster"] < 0, np.nan, z["qcluster"]).astype(float), time=qtime,
                        lead_h=(qtime - issue).astype(float) / 3600.0, lat=z["qlat"].astype(float), lon=z["qlon"].astype(float))
            for j, v in enumerate(VARS):
                ok = np.isfinite(truth[:, j])
                frames.append(pd.DataFrame({**{k: (a[ok] if isinstance(a, np.ndarray) else a) for k, a in base.items()},
                                            "var": v, "truth": truth[ok, j].astype(float), "pred": pred[ok, j].astype(float), "sigma": sigma[ok, j].astype(float)}))
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--samples", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="model")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu; cuda fails loudly if unavailable")
    ap.add_argument("--drop-iono", action="store_true"); ap.add_argument("--drop-glotec", action="store_true"); ap.add_argument("--drop-spots", action="store_true")
    ap.add_argument("--start", help="only samples with file name >= YYYYMMDD"); ap.add_argument("--end")
    ap.add_argument("--max-tok", type=int, help="override the checkpoint's ionosonde token budget (default: as trained)"); ap.add_argument("--recency-tau", type=float)
    ap.add_argument("--spot-within", help="lat,lon,km: keep only spot tokens within km of the point (single-receiver study)")
    a = ap.parse_args()
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False (driver/wheel mismatch?)")
    ck_args = torch.load(a.ckpt, map_location="cpu", weights_only=False)["args"]
    df = predict_dir(load(a.ckpt, dev), a.samples, a.name, dev, drop_iono=a.drop_iono, drop_glotec=a.drop_glotec, start=a.start, end=a.end, drop_spots=a.drop_spots,
                     max_tok=a.max_tok or ck_args.get("max_tok"), recency_tau=a.recency_tau or ck_args.get("recency_tau"),
                     spot_within=tuple(float(x) for x in a.spot_within.split(",")) if a.spot_within else None)
    df.to_parquet(a.out)
    print(f"{len(df):,} rows -> {a.out}")


if __name__ == "__main__":
    main()
