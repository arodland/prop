"""Learned relevance: where the decoder's cross-attention mass goes, by token kind, binned by lead, query kind
and distance to the nearest ionosonde token. Complements the source-dropout ablation (predict.py --drop-*),
which measures skill lost; this measures what the model looked at.

    uv run --extra train analysis/attention_mass.py --ckpt /kass/forecast/models/v5_spots/best.pt \
        --samples /kass/forecast/samples/val_v5 --out /kass/forecast/eval/attn_v5.parquet [--limit N]

Output: one row per (sample, query) with mass per kind (null, global, iono, glotec, spot) summing to 1, plus
mass per *token* for each kind (mass / token count, so a kind with 3000 tokens is comparable to one with 50);
prints a table by lead bucket and by distance bucket.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
sys.path.insert(0, os.path.dirname(__file__))
from predict import load  # noqa: E402
from train import pad_tok  # noqa: E402
from skill_by_lead_distance import DIST_EDGES, DIST_LABELS, gc_km  # noqa: E402

KINDS = ("null", "global", "iono", "glotec", "spot")
LEAD_EDGES = [0, 1, 3, 6, 12, 24]; LEAD_LABELS = ["0–1", "1–3", "3–6", "6–12", "12–24"]


def sample_mass(model, z, dev, bs_q=2048):
    tok, tg, ts = pad_tok(z["tok"]), z["tok_g"] if "tok_g" in z.files else np.zeros((0, 14), np.float32), z["tok_s"] if "tok_s" in z.files else np.zeros((0, model.f_spot), np.float32)
    tg = tg[tg[:, 7] < 0.5]  # no qf=0 cells
    if not model.glotec:
        tg = tg[:0]
    if not model.spots:
        ts = ts[:0]
    t = lambda a: torch.from_numpy(a)[None].to(dev)
    ones = lambda n: torch.ones(1, n, dtype=torch.bool, device=dev)
    h, pad = model.encode(t(tok), ones(len(tok)), t(np.nan_to_num(z["glob"])), t(tg), ones(len(tg)), t(ts), ones(len(ts)))
    # encode() column order: null, global, iono..., glotec..., spot...
    bounds = np.cumsum([0, 1, 1, len(tok), len(tg), len(ts)])
    ws = []
    for i in range(0, len(z["qry"]), bs_q):
        ws.append(model.decode(t(z["qry"][i:i + bs_q]), h, pad, return_attn=True)[2][0].cpu().numpy())
    w = np.concatenate(ws)  # (M, N')
    mass = {k: w[:, bounds[j]:bounds[j + 1]].sum(1) for j, k in enumerate(KINDS)}
    counts = {k: bounds[j + 1] - bounds[j] for j, k in enumerate(KINDS)}
    # distance from each query to the nearest ionosonde token (tokens carry lat/90, sin lon, cos lon)
    tlat, tlon = tok[:, 0] * 90, np.degrees(np.arctan2(tok[:, 1], tok[:, 2]))
    st = np.unique(np.round(np.c_[tlat, tlon], 2), axis=0)
    d = gc_km(z["qlat"][:, None], z["qlon"][:, None], st[None, :, 0], st[None, :, 1]).min(1) if len(st) else np.full(len(z["qlat"]), np.inf)
    issue = np.datetime64(int(z["issue"]), "s")
    df = pd.DataFrame({"issue_time": issue, "lead_h": (z["qtime"].astype("datetime64[s]") - issue).astype(float) / 3600, "qkind": np.where(z["qkind"] == 1, "ro", "iono"),
                       "held": z["held"], "dist_km": d, **{f"m_{k}": mass[k] for k in KINDS}, **{f"n_{k}": counts[k] for k in KINDS}})
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--samples", required=True); ap.add_argument("--out")
    ap.add_argument("--limit", type=int, help="first N samples only"); ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model = load(a.ckpt, dev)
    files = sorted(glob.glob(f"{a.samples}/*.npz"))[: a.limit]
    with torch.no_grad():
        df = pd.concat([sample_mass(model, np.load(f), dev) for f in files], ignore_index=True)
    if a.out:
        df.to_parquet(a.out)
    df["lead"] = pd.cut(df.lead_h, LEAD_EDGES, labels=LEAD_LABELS); df["dist"] = pd.cut(df.dist_km, DIST_EDGES, labels=DIST_LABELS)
    mcols = [f"m_{k}" for k in KINDS]
    pd.set_option("display.width", 200); pd.set_option("display.float_format", "{:.3f}".format)
    print(f"{len(files)} samples, {len(df):,} queries; tokens per kind (mean): " + ", ".join(f"{k} {df[f'n_{k}'].mean():.0f}" for k in KINDS))
    print("\nmass by lead bucket"); print(df.groupby("lead", observed=True)[mcols].mean())
    print("\nmass by distance to nearest ionosonde token (ionosonde + RO queries)"); print(df.groupby("dist", observed=True)[mcols].mean())
    print("\nmass by query kind"); print(df.groupby("qkind")[mcols].mean())
    per_tok = pd.DataFrame({k: df[f"m_{k}"] / df[f"n_{k}"].clip(lower=1) for k in KINDS if df[f"n_{k}"].max() > 0})
    print("\nmass per token (mean over queries), relative to iono"); print((per_tok.mean() / per_tok["iono"].mean()).to_string())


if __name__ == "__main__":
    main()
