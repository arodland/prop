"""Rolling-origin evaluation, and the production update step (PLAN.md Phase 5): for each month M,
fine-tune the current checkpoint on samples with issue time < M, score month M out of sample.

    uv run --extra train train/rolling.py --base models/v2/best.pt --samples samples/eval2025 --history samples/train \
        --months 2025-01 2025-12 --out models/v2_rolling [--steps 300 --lr 2e-5 --window-months 6 --from-base]

Without --history the trailing window only sees the eval directory (a few weeks in the first months), and
2000 steps on a few hundred samples overfits badly (first run: RMSE rose month over month).

Writes <out>/<M>.pt and <out>/<M>.parquet (replay schema, model name 'model_rolling'); concatenate the parquet
files for eval/report.py. Fine-tuning is chained month to month (M's checkpoint starts from M-1's), which is
exactly what the production job does with each new month of data.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from model import AnomalyModel, gaussian_nll  # noqa: E402
from predict import predict_dir  # noqa: E402
from train import Samples, collate  # noqa: E402


class Subset(Samples):
    def __init__(self, files, train):
        self.files = files; self.train = train; self.max_q = 1024


def month_of(f):
    return os.path.basename(f)[:6]  # YYYYMM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--samples", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--months", nargs=2, required=True, help="first and last month to score, YYYY-MM")
    ap.add_argument("--steps", type=int, default=300, help="fine-tune steps per month (~1.5 epochs of a 6-month window at bs 8)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--window-months", type=int, default=6); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--history", nargs="*", default=[], help="extra sample dirs (e.g. the training set) so the trailing window is full from the first month")
    ap.add_argument("--from-base", action="store_true", help="fine-tune from the base checkpoint each month instead of chaining month to month")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu; cuda fails loudly if unavailable")
    a = ap.parse_args()
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False (driver/wheel mismatch?)")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(a.base, map_location=dev)
    model = AnomalyModel(d=ck["args"]["d"], enc_layers=ck["args"]["layers"], query_self_attn=ck["args"].get("query_self_attn", True), glotec=ck["args"].get("glotec", False), spots=ck["args"].get("spots", False), f_spot=ck["args"].get("f_spot", 29), f_qry=ck["args"].get("f_qry", 10)).to(dev); model.load_state_dict(ck["model"])
    files = sorted(set(glob.glob(f"{a.samples}/*.npz")) | {f for h in a.history for f in glob.glob(f"{h}/*.npz")})
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    months = pd.period_range(a.months[0], a.months[1], freq="M")
    for M in months:
        m_str = M.strftime("%Y%m")
        lo = (M - a.window_months).strftime("%Y%m")
        train_files = [f for f in files if lo <= month_of(f) < m_str]
        score_files = [f for f in files if month_of(f) == m_str]
        if not score_files:
            print(f"{M}: no samples, skip"); continue
        if a.from_base:
            model.load_state_dict(base_state)
        tmp = out / f"_score_{m_str}"; tmp.mkdir(exist_ok=True)
        for f in score_files:
            dst = tmp / os.path.basename(f)
            if not dst.exists():
                os.symlink(f, dst)
        model.eval(); pre = predict_dir(model, str(tmp), "model_pre", dev)  # the un-tuned model on this month, for reference
        ho = pre[(pre["mode"] == "holdout") & (pre["var"] == "fof2")]; pre_rmse = np.sqrt(((ho.pred - ho.truth) ** 2).mean())
        if train_files:
            dl = DataLoader(Subset(train_files, True), a.bs, shuffle=True, collate_fn=collate, num_workers=a.workers, drop_last=True)
            opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.01); model.train(); step = 0
            while step < a.steps:
                for tok, tmask, g, qry, tgt, held, kind, tok_g, g_mask, tok_s, s_mask in dl:
                    tok, tmask, g, qry, tgt, held = tok.to(dev), tmask.to(dev), g.to(dev), qry.to(dev), tgt.to(dev), held.to(dev)
                    mean, logvar = model(tok, tmask, g, qry, tok_g.to(dev), g_mask.to(dev), tok_s.to(dev), s_mask.to(dev))
                    loss, _ = gaussian_nll(mean, logvar, tgt, weight=1.0 + held.float())
                    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
                    if step >= a.steps:
                        break
        model.eval()
        torch.save({"model": model.state_dict(), "args": ck["args"], "month": m_str}, out / f"{m_str}.pt")
        df = predict_dir(model, str(tmp), "model_rolling", dev)
        df.to_parquet(out / f"{m_str}.parquet")
        ho = df[(df["mode"] == "holdout") & (df["var"] == "fof2")]
        print(f"{M}: fine-tuned on {len(train_files)} samples ({a.steps} steps), scored {len(score_files)}; held-out foF2 RMSE {np.sqrt(((ho.pred - ho.truth) ** 2).mean()):.3f}  (before fine-tune: {pre_rmse:.3f})", flush=True)


if __name__ == "__main__":
    main()
