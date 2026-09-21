"""Train the anomaly model on prebuilt samples.

    uv run --extra train train/train.py --train /kass/forecast/samples/train --val /kass/forecast/samples/val --out /kass/forecast/models/v0

Source dropout (PLAN.md): with prob P_DROP_GLOB the global token is zeroed; station dropout: each sample
keeps a random 50-100% of its tokens. Loss: Gaussian NLL on normalised anomalies, held-out queries
weighted x2 (they are the primary metric). Val metric: RMSE (in MHz) on held-out ionosonde fof2.
"""
import argparse
import copy
import glob
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import AnomalyModel, F_QRY, gaussian_nll

ANOM_SCALE = np.array([1.5, 40.0, 4.5], np.float32)
P_DROP_GLOB = 0.15
P_DROP_GLO = 0.3  # source dropout: drop all GloTEC tokens for this sample
P_DROP_SPOT = 0.3


def pad_tok(tok):
    """v1..v5 samples have 16 token features; F_TOK is now 17 (log1p(rows pooled)/3, = log(2)/3 for an unpooled row)."""
    return tok if tok.shape[1] >= 17 else np.c_[tok, np.full(len(tok), np.log1p(1) / 3, np.float32)]


def subsample(tok, n, rng, tau_h=None):
    """Keep n of the ionosonde tokens. Uniform (as build_samples did at 2048), or recency-weighted with
    P(keep) ∝ exp(age/τ) where age = tok[:, 5]*24 h (negative), so the last few minutes at busy stations survive."""
    if len(tok) <= n:
        return tok
    if tau_h:
        w = np.exp(tok[:, 5] * 24.0 / tau_h); w /= w.sum()
        return tok[rng.choice(len(tok), n, replace=False, p=w)]
    return tok[rng.choice(len(tok), n, replace=False)]


def spatial_cap(tok, tok_g, tok_s, qry, issue, r_lo, r_hi, rng):
    """Spatial dropout (2026-09-17): keep only the observation tokens inside a random cap, so "no observations here"
    is a training case. Centre = a random token of any kind (caps land where data is), radius ~ U(r_lo, r_hi) km.
    Queries are untouched; the nearest-station query state (columns F_QRY:) is recomputed from the kept ionosonde
    rows so nothing outside the cap leaks through it. Token geometry: cols 0 lat/90, 1 sin lon, 2 cos lon; ionosonde
    tokens carry dt/24 (col 5, negative hours) and the normalised anomaly in cols 9:12."""
    pools = [t for t in (tok, tok_g, tok_s) if len(t)]
    if not pools:
        return tok, tok_g, tok_s, qry
    allg = np.concatenate([t[:, :3] for t in pools]); c = allg[rng.integers(len(allg))]
    clat, clon = np.radians(c[0] * 90), np.arctan2(c[1], c[2]); r = rng.uniform(r_lo, r_hi)
    def inside(t):
        if not len(t):
            return t
        la, lo = np.radians(t[:, 0] * 90), np.arctan2(t[:, 1], t[:, 2])
        d = 6371 * np.arccos(np.clip(np.sin(la) * np.sin(clat) + np.cos(la) * np.cos(clat) * np.cos(lo - clon), -1, 1))
        return t[d <= r]
    tok, tok_g, tok_s = inside(tok), inside(tok_g), inside(tok_s)
    if qry.shape[1] > F_QRY:
        from build_samples import query_state
        t0 = np.datetime64(int(issue), "s")
        qry = qry.copy()
        if len(tok):
            s_time = t0 + (tok[:, 5] * 24 * 3600).astype("timedelta64[s]")
            qry[:, F_QRY:] = query_state(tok[:, 0] * 90, np.degrees(np.arctan2(tok[:, 1], tok[:, 2])), s_time, tok[:, 9:12], t0,
                                          qry[:, 0] * 90, np.degrees(np.arctan2(qry[:, 1], qry[:, 2])))
        else:
            qry[:, F_QRY:] = 0; qry[:, F_QRY + 1] = 1.0
    return tok, tok_g, tok_s, qry


class Samples(Dataset):
    def __init__(self, d, train, max_q=1024, max_tok=None, recency_tau=None, p_drop_iono=0.0, p_drop_glotec=P_DROP_GLO, p_drop_spots=P_DROP_SPOT, p_drop_iono_glotec=None, lead_window=0.0, p_spatial=0.0, spatial_km=(1500.0, 6000.0)):
        self.files = sorted(glob.glob(f"{d}/*.npz")); self.train = train; self.max_q = max_q
        self.max_tok = max_tok; self.recency_tau = recency_tau  # token-budget study: applied at load time, train and val alike
        # whole-source dropout. Ionosondes were never dropped whole before 2026-09-08 (only 50-100% token keep), so the
        # spots-only / GloTEC-only / no-input modes were never trained on directly and converge late and unevenly.
        self.p_drop_iono, self.p_drop_glotec, self.p_drop_spots = p_drop_iono, p_drop_glotec, p_drop_spots
        self.lead_window = lead_window
        self.p_spatial, self.spatial_km = p_spatial, spatial_km
        # GloTEC exists only from 2025-05, i.e. ~7% of training samples, so the GloTEC-only mode gets ~0.5% of the
        # steps spots-only gets; a higher ionosonde-drop rate on GloTEC-era samples evens that up.
        self.p_drop_iono_glotec = p_drop_iono if p_drop_iono_glotec is None else p_drop_iono_glotec

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        z = np.load(self.files[i])
        tok, qry, tgt, held = pad_tok(z["tok"]), z["qry"], z["tgt"], z["held"]
        glob_ = np.nan_to_num(z["glob"].copy())  # defensive: index gaps must never poison a batch
        if self.max_tok:
            tok = subsample(tok, self.max_tok, np.random.default_rng(np.random.randint(2**31) if self.train else int(z["issue"])), self.recency_tau)
        tok_g_all = z["tok_g"] if "tok_g" in z.files else np.zeros((0, 14), np.float32)
        tok_g_all = tok_g_all[tok_g_all[:, 7] < 0.5]  # drop qf=0 cells (one-hot column 7) from samples built before 2026-09-09; new builds have none
        tok_s_all = z["tok_s"] if "tok_s" in z.files else np.zeros((0, 29), np.float32)  # width only matters when present
        if self.train and self.p_spatial > 0 and np.random.rand() < self.p_spatial:
            tok, tok_g_all, tok_s_all, qry = spatial_cap(tok, tok_g_all, tok_s_all, qry, z["issue"], self.spatial_km[0], self.spatial_km[1], np.random.default_rng(np.random.randint(2**31)))
        if self.train:
            has_g = len(tok_g_all) > 0
            if np.random.rand() < (self.p_drop_iono_glotec if has_g else self.p_drop_iono):
                tok = tok[:0]  # whole-source drop: the model must answer from GloTEC / spots / indices alone
                qry = qry.copy(); qry[:, F_QRY:] = 0  # the nearest-station state (--qstate columns) is ionosonde data too
            else:
                keep = np.random.rand(len(tok)) < np.random.uniform(0.5, 1.0)
                tok = tok[keep] if keep.sum() >= 5 else tok
            if np.random.rand() < P_DROP_GLOB:
                glob_[:] = 0
            if self.lead_window:  # anti-memorisation: each target hour is covered by 8 issue times a day; keeping one random
                k0 = np.random.uniform(0, 24 - self.lead_window)  # lead window per sample per epoch shows each target ~once per epoch
                lead_h = qry[:, 5] * 24.0; keep_q = (lead_h >= k0) & (lead_h < k0 + self.lead_window)
                if keep_q.sum() >= 32:
                    qry, tgt, held = qry[keep_q], tgt[keep_q], held[keep_q]
            if len(qry) > self.max_q:
                sel = np.random.choice(len(qry), self.max_q, replace=False); qry, tgt, held = qry[sel], tgt[sel], held[sel]
        tok_g = tok_g_all
        if self.train and np.random.rand() < self.p_drop_glotec:
            tok_g = tok_g[:0]
        elif self.train and len(tok_g):
            tok_g = tok_g[np.random.rand(len(tok_g)) < np.random.uniform(0.5, 1.0)]  # per-token keep (v5b): fewer memorisable fingerprints
            # Δt jitter (±20 min) so the model sees lags densely, not only at the builder's fixed slots;
            # in production the tokens carry whatever true Δt the available steps have.
            tok_g = tok_g.copy(); tok_g[:, 5] += np.random.uniform(-20, 20, len(tok_g)).astype(np.float32) / 60 / 24
        tok_s = tok_s_all
        if self.train and np.random.rand() < self.p_drop_spots:
            tok_s = tok_s[:0]
        elif self.train and len(tok_s):
            tok_s = tok_s[np.random.rand(len(tok_s)) < np.random.uniform(0.5, 1.0)]
        return tok, glob_, qry, tgt, held, z["qkind"] if len(z["qkind"]) == len(qry) else np.zeros(len(qry), np.int8), tok_g, tok_s


def _worker_init(_):
    np.random.seed(torch.initial_seed() % 2**32)  # each worker/epoch gets its own numpy stream, derived from the (seeded) torch generator


def collate(batch):
    B = len(batch); N = max(len(b[0]) for b in batch); M = max(len(b[2]) for b in batch); Gn = max(len(b[6]) for b in batch); Sn = max(len(b[7]) for b in batch)
    tok_g = torch.zeros(B, Gn, 14); g_mask = torch.zeros(B, Gn, dtype=torch.bool); tok_s = torch.zeros(B, Sn, max(b[7].shape[1] for b in batch)); s_mask = torch.zeros(B, Sn, dtype=torch.bool)
    tok = torch.zeros(B, N, batch[0][0].shape[1]); tmask = torch.zeros(B, N, dtype=torch.bool)
    qry = torch.zeros(B, M, batch[0][2].shape[1]); tgt = torch.full((B, M, 3), float("nan")); held = torch.zeros(B, M, dtype=torch.bool); kind = torch.zeros(B, M, dtype=torch.int8)
    glob_ = torch.zeros(B, batch[0][1].shape[0])
    for i, (t, g, q, y, h, k, tg, tsp) in enumerate(batch):
        tok[i, :len(t)] = torch.from_numpy(t); tmask[i, :len(t)] = True; glob_[i] = torch.from_numpy(g)
        if len(tg):
            tok_g[i, :len(tg)] = torch.from_numpy(tg); g_mask[i, :len(tg)] = True
        if len(tsp):
            tok_s[i, :len(tsp)] = torch.from_numpy(tsp); s_mask[i, :len(tsp)] = True
        qry[i, :len(q)] = torch.from_numpy(q); tgt[i, :len(y)] = torch.from_numpy(y); held[i, :len(h)] = torch.from_numpy(h); kind[i, :len(k)] = torch.from_numpy(k.astype(np.int8))
    return tok, tmask, glob_, qry, tgt, held, kind, tok_g, g_mask, tok_s, s_mask


def evaluate(model, dl, dev, chunk=1024, drop=()):
    # Val samples carry every query (~9k) and every token; the decoder is cross-attention only, so
    # chunking the queries gives identical output at 1/9 the peak memory (B*heads*M*N attention scores).
    # drop: subset of {"iono", "glotec", "spots"} removed at evaluation time (the outage / single-source modes).
    model.eval(); se = torch.zeros(3); n = torch.zeros(3); se_iri = torch.zeros(3)
    with torch.no_grad():
        for tok, tmask, g, qry, tgt, held, kind, tok_g, g_mask, tok_s, s_mask in dl:
            if "iono" in drop:
                tok, tmask = tok[:, :0], tmask[:, :0]; qry = qry.clone(); qry[..., F_QRY:] = 0  # and the query-state columns
            if "glotec" in drop:
                tok_g, g_mask = tok_g[:, :0], g_mask[:, :0]
            if "spots" in drop:
                tok_s, s_mask = tok_s[:, :0], s_mask[:, :0]
            tok, tmask, g, qry, tgt = tok.to(dev), tmask.to(dev), g.to(dev), qry.to(dev), tgt.to(dev)
            tok_g, g_mask, tok_s, s_mask = tok_g.to(dev), g_mask.to(dev), tok_s.to(dev), s_mask.to(dev)
            h, pad = model.encode(tok, tmask, g, tok_g, g_mask, tok_s, s_mask)
            mean = torch.cat([model.decode(qry[:, i:i + chunk], h, pad)[0] for i in range(0, qry.shape[1], chunk)], 1)
            m = torch.isfinite(tgt) & held.to(dev)[..., None] & (kind.to(dev) == 0)[..., None]  # held-out ionosonde queries
            t = torch.where(m, tgt, torch.zeros_like(tgt))
            se += (((t - mean) ** 2) * m).sum((0, 1)).cpu(); se_iri += ((t ** 2) * m).sum((0, 1)).cpu(); n += m.sum((0, 1)).cpu()
    model.train()
    return (se / n).sqrt().numpy() * ANOM_SCALE, (se_iri / n).sqrt().numpy() * ANOM_SCALE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--val", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--layers", type=int, default=4); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--glotec", action="store_true", help="enable the GloTEC token pathway (samples must carry tok_g)")
    ap.add_argument("--spots", action="store_true", help="enable the spot-activity token pathway (samples must carry tok_s)")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu; cuda fails loudly if unavailable")
    ap.add_argument("--p-drop-iono", type=float, default=0.0, help="whole-source dropout of the ionosonde tokens (default 0 = the v1..v7 recipe; try 0.2)")
    ap.add_argument("--p-drop-glotec", type=float, default=P_DROP_GLO); ap.add_argument("--p-drop-spots", type=float, default=P_DROP_SPOT)
    ap.add_argument("--curriculum", type=int, default=0, help="epochs over which the whole-source drop rates anneal linearly from --curriculum-start to their targets (0 = off). Early epochs are then mostly single-source samples, so each pathway learns to predict on its own before the model learns to combine them")
    ap.add_argument("--curriculum-start", default="0.6,0.7,0.7", help="starting drop rates iono,glotec,spots (P(all three present) = 3.6%% vs 39%% at the 0.2/0.3/0.3 targets)")
    ap.add_argument("--p-drop-iono-glotec", type=float, help="ionosonde whole-source drop rate on samples that have GloTEC tokens (default = --p-drop-iono; try 0.5)")
    ap.add_argument("--dropout", type=float, default=0.1, help="attention/FFN dropout in every layer (overfitting knob; v1..v7 used 0.1)")
    ap.add_argument("--wd", type=float, default=0.01, help="AdamW / Muon decoupled weight decay (v1..v7 used 0.01)")
    ap.add_argument("--short-lead-weight", type=float, default=1.0, help="loss weight multiplier for queries with lead <= --short-lead-h (1 = off). Short leads are 1/24 of the queries but hold the largest known loss (own-station 0-4 h); 3 is a sensible first try")
    ap.add_argument("--short-lead-h", type=float, default=3.0)
    ap.add_argument("--val-modes", action="store_true", help="also print per-epoch val foF2 with only ionosondes / only GloTEC / only spots / no inputs (4 extra val passes)")
    ap.add_argument("--p-spatial", type=float, default=0.0, help="spatial dropout: fraction of training samples reduced to the observation tokens inside a random cap (see spatial_cap); 0.3 is the first try")
    ap.add_argument("--spatial-km", default="1500,6000", help="cap radius range in km for --p-spatial")
    ap.add_argument("--seed", type=int, default=None, help="seed torch/numpy and the DataLoader (worker augmentation varies per epoch but is reproducible)")
    ap.add_argument("--lead-window", type=float, default=0.0, help="hours; in training keep only the targets inside one random lead window per sample per epoch (0 = all leads). With 3-h issue times each target hour is in 8 samples; a 3-h window shows it ~once per epoch, which removes the repeated exposure that drives memorisation (target noise cannot: the clean target is recoverable by averaging)")
    ap.add_argument("--target-noise", type=float, default=0.0, help="Gaussian noise added to the normalised targets each step (0.15 ≈ 0.2 MHz foF2, the autoscaling error); makes memorisation impossible so more epochs stay useful")
    ap.add_argument("--logvar-floor", type=float, default=None, help="floor on the predicted log-variance in the loss (normalised units; -4 ≈ σ 0.2 MHz foF2): stops the NLL going negative by σ collapse on memorised points")
    ap.add_argument("--keep-epochs", action="store_true", help="also save the EMA weights after every epoch (ema_epNN.pt, ~5 MB each)")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay for the weights (0 = off; 0.999 recommended). best.pt = best EMA, best_raw.pt = best raw")
    ap.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw", help="muon = orthogonalised momentum on weight matrices (train/muon.py, from SSTVAE); same --lr and schedule")
    ap.add_argument("--muon-adjust-lr-fn", default="match_rms_adamw", choices=["match_rms_adamw", "original", "none"])
    ap.add_argument("--max-tok", type=int, help="ionosonde token budget applied at load time (token-budget study; samples built with --max-tok 8192)")
    ap.add_argument("--recency-tau", type=float, help="hours; with --max-tok, keep tokens with P ∝ exp(age/τ) instead of uniformly")
    a = ap.parse_args()
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False (driver/wheel mismatch?)")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gen = None
    if a.seed is not None:  # reproducible runs: the 1.010 of v9b was a lucky draw, 1.03-1.04 is typical for the recipe
        torch.manual_seed(a.seed); np.random.seed(a.seed); gen = torch.Generator().manual_seed(a.seed)
    tr = DataLoader(Samples(a.train, True, max_tok=a.max_tok, recency_tau=a.recency_tau, p_drop_iono=a.p_drop_iono, p_drop_glotec=a.p_drop_glotec, p_drop_spots=a.p_drop_spots, p_drop_iono_glotec=a.p_drop_iono_glotec, lead_window=a.lead_window, p_spatial=a.p_spatial, spatial_km=tuple(float(x) for x in a.spatial_km.split(","))),
                    a.bs, shuffle=True, collate_fn=collate, num_workers=a.workers, drop_last=True, generator=gen, worker_init_fn=_worker_init)
    va = DataLoader(Samples(a.val, False, max_tok=a.max_tok, recency_tau=a.recency_tau), a.bs, shuffle=False, collate_fn=collate, num_workers=a.workers)
    z0 = np.load(tr.dataset.files[0]); f_spot = int(z0["tok_s"].shape[1]) if a.spots else 29  # token format is set by the samples
    pool = bool(z0["pool"]) if "pool" in z0.files else False  # pooled ionosonde tokens (build_samples --pool); the service must match
    spot_res = int(z0["spot_res"]) if "spot_res" in z0.files else 60
    spot_baseline = str(z0["spot_baseline"]) if "spot_baseline" in z0.files else "frozen"  # baseline scheme of the samples
    f_qry = int(z0["qry"].shape[1])  # 18 with --qstate samples  # spot bin resolution the samples were built with (SPOT_RES for the service)
    model = AnomalyModel(d=a.d, enc_layers=a.layers, dropout=a.dropout, query_self_attn=False, glotec=a.glotec, spots=a.spots, f_spot=f_spot, f_qry=f_qry).to(dev)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, {len(tr.dataset)} train / {len(va.dataset)} val samples, device {dev}")
    if a.optimizer == "muon":
        # Muon on the weight matrices, AdamW on biases/LayerNorm gains and on the embedding-like tensors (null token,
        # kind embeddings) and the output head, per the usual Muon guidance. match_rms_adamw scaling keeps --lr's meaning.
        from muon import Muon
        skip = {"null", "kind_emb", "head.3.weight"}
        mats = [p for n, p in model.named_parameters() if p.ndim >= 2 and n not in skip]
        rest = [p for n, p in model.named_parameters() if p.ndim < 2 or n in skip]
        opt = Muon([{"params": mats, "use_muon": True, "lr": a.lr, "weight_decay": a.wd},
                    {"params": rest, "use_muon": False, "lr": a.lr, "weight_decay": a.wd}], lr=a.lr, adjust_lr_fn=a.muon_adjust_lr_fn)
        print(f"muon: {sum(p.numel() for p in mats) / 1e6:.2f}M params orthogonalised, {sum(p.numel() for p in rest) / 1e3:.0f}k on AdamW", flush=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=a.wd)
    steps = a.epochs * len(tr); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 500) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    # EMA of the weights (--ema 0.999 ≈ a 1000-step ≈ half-epoch average): smooths the epoch-boundary lottery in the
    # val numbers; decay ramps up from 0 so the first steps are not anchored to the init. Selection uses the EMA model.
    ema = copy.deepcopy(model).eval() if a.ema > 0 else None
    for p_ in (ema.parameters() if ema else []):
        p_.requires_grad_(False)
    args = {**vars(a), "query_self_attn": False, "glotec": a.glotec, "spots": a.spots, "f_spot": f_spot, "pool": pool, "spot_res": spot_res, "spot_baseline": spot_baseline, "f_qry": f_qry}
    best = best_raw = float("inf"); step = 0
    targets = (a.p_drop_iono, a.p_drop_glotec, a.p_drop_spots, tr.dataset.p_drop_iono_glotec)
    starts = [float(x) for x in a.curriculum_start.split(",")]; starts.append(max(starts[0], targets[3]))
    for ep in range(a.epochs):
        t0 = time.time(); tot = 0.0
        if a.curriculum:  # workers are re-forked each epoch (persistent_workers=False), so dataset attributes set here apply
            w = min(1.0, ep / a.curriculum)
            tr.dataset.p_drop_iono, tr.dataset.p_drop_glotec, tr.dataset.p_drop_spots, tr.dataset.p_drop_iono_glotec = [s0 + w * (t - s0) for s0, t in zip(starts, targets)]
            print(f"curriculum epoch {ep + 1}: drop rates iono {tr.dataset.p_drop_iono:.2f} glotec {tr.dataset.p_drop_glotec:.2f} spots {tr.dataset.p_drop_spots:.2f} iono|glotec {tr.dataset.p_drop_iono_glotec:.2f}", flush=True)
        for tok, tmask, g, qry, tgt, held, kind, tok_g, g_mask, tok_s, s_mask in tr:
            tok, tmask, g, qry, tgt, held = tok.to(dev), tmask.to(dev), g.to(dev), qry.to(dev), tgt.to(dev), held.to(dev)
            mean, logvar = model(tok, tmask, g, qry, tok_g.to(dev), g_mask.to(dev), tok_s.to(dev), s_mask.to(dev))
            w = 1.0 + held.float()
            if a.short_lead_weight != 1.0:  # short leads are 1/24 of the queries; the 0-4 h gap to the GP lives there
                w = w * torch.where(qry[..., 5] * 24.0 <= a.short_lead_h, a.short_lead_weight, 1.0)
            if a.target_noise > 0:  # anti-memorisation: targets carry autoscaling error anyway; noise at that level makes exact fits impossible
                tgt = tgt + a.target_noise * torch.randn_like(tgt)
            if a.logvar_floor is not None:  # and the NLL cannot be driven negative by collapsing σ on memorised points
                logvar = logvar.clamp(min=a.logvar_floor)
            loss, _ = gaussian_nll(mean, logvar, tgt, weight=w)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tot += loss.item(); step += 1
            if ema is not None:
                d = min(a.ema, (1 + step) / (10 + step))
                with torch.no_grad():
                    torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - d)
        rmse, rmse_iri = evaluate(model, va, dev)
        line = f"epoch {ep + 1}: loss {tot / len(tr):.4f}  val held-out RMSE fof2 {rmse[0]:.3f} hmf2 {rmse[1]:.1f} mufd {rmse[2]:.2f}"
        if ema is not None:
            rmse_e, _ = evaluate(ema, va, dev); line += f"  | EMA fof2 {rmse_e[0]:.3f} hmf2 {rmse_e[1]:.1f} mufd {rmse_e[2]:.2f}"
            if a.keep_epochs:  # every EMA checkpoint, so a later pass can pick per mode or average epochs (single-source modes peak earlier than the primary)
                torch.save({"model": ema.state_dict(), "args": args, "val_rmse": rmse_e.tolist(), "epoch": ep + 1}, out / f"ema_ep{ep + 1:02d}.pt")
            if rmse_e[0] < best:
                best = rmse_e[0]; torch.save({"model": ema.state_dict(), "args": args, "val_rmse": rmse_e.tolist(), "epoch": ep + 1}, out / "best.pt")
            if rmse[0] < best_raw:
                best_raw = rmse[0]; torch.save({"model": model.state_dict(), "args": args, "val_rmse": rmse.tolist(), "epoch": ep + 1}, out / "best_raw.pt")
        elif rmse[0] < best:
            best = rmse[0]; torch.save({"model": model.state_dict(), "args": args, "val_rmse": rmse.tolist(), "epoch": ep + 1}, out / "best.pt")
        if a.val_modes:  # single-source / no-input foF2 on the same val rows, from the model that best.pt would hold
            sel = ema if ema is not None else model
            modes = {"iono": ("glotec", "spots"), "glotec": ("iono", "spots"), "spots": ("iono", "glotec"), "none": ("iono", "glotec", "spots")}
            modes = {k: d for k, d in modes.items() if (a.glotec or k != "glotec") and (a.spots or k != "spots")}  # only pathways the model has
            line += "  | only:" + " ".join(f"{k} {evaluate(sel, va, dev, drop=d)[0][0]:.3f}" for k, d in modes.items())
        print(f"{line}   (IRI: {rmse_iri[0]:.3f} / {rmse_iri[1]:.1f} / {rmse_iri[2]:.2f})  {time.time() - t0:.0f}s", flush=True)
    print("best val fof2 RMSE", best, "(EMA; raw best %.3f in best_raw.pt)" % best_raw if ema is not None else "")


if __name__ == "__main__":
    main()
