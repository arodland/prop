# Anomaly forecast model, architecture (Phase 4a/4b)

TLDR: predict the anomaly on top of cached IRI with a small transformer over observation tokens,
queried at any point; Gaussian NLL, dropout at token and source level, rolling monthly fine-tunes.
Bar to clear: production GP.

Companion to `../PLAN.md`. This describes what `train/` builds and why. A rendered version with the
data-flow diagram is published as an artifact ("Anomaly Model Architecture").

## 1. What is predicted

The model never predicts foF2 directly. It predicts the **anomaly** `a = obs − IRI` at any
(lat, lon, t), and the forecast is `IRI(lat, lon, t) + a`. IRI (PyIRI, SHU2015 hmF2, trailing
81-day F10.7 driver, served from the map cache) supplies the diurnal, seasonal and solar-cycle
structure and the smooth global mean; the model only has to learn how today departs from it and
how that departure moves. Three variables share one network: foF2 [MHz], hmF2 [km], MUF(3000)
[MHz], normalised by (1.5, 40, 4.5) so each anomaly is O(1).

Why this and not the previous generative design: the metric is point RMSE, the 22M real
observations are the supervision, and a conditional mean with a variance is exactly what the site
and ITURHFPROP consume. See PLAN.md Phase 4 for the full argument.

## 2. Data flow

```
 ionosondes (T-24h..T-15m) ─┐
 GloTEC cells (4b)          ├─► tokens ──► encoder (self-attn ×4) ──► memory
 spot cell anomalies (4b)   │      ▲                                    │
 indices (global token)    ─┘      │ null token (always present)         │ cross-attn ×2
                                                                        ▼
 queries: any (lat, lon, t) in (T, T+24h] ───────────────────────► decoder ──► heads
   (station rows for training; the 361×181 grid × 25 leads for maps)        │
                                                                        ▼
                                  IRI(lat, lon, t) + μ_anomaly ,  σ_anomaly
```

One sample = one issue time T. Everything the model sees is strictly before T − 15 min (the
assumed arrival latency); everything it is scored on is after T.

## 3. Tokens

Every input is a token in the same d=128 space, so adding a source is adding an input projection.

| token kind | one token per | features | status |
|---|---|---|---|
| ionosonde | observation row | lat/90 · sin,cos lon · sin,cos local time · Δt/24 · IRI(foF2,hmF2,MUF)/scale · anomaly×3 · present×3 · confidence | built |
| global | issue time | F10.7₈₁/200 · F10.7_daily/200 · log ap(T)/5 · log max₂₄ap/5 · sin,cos day-of-year | built |
| null | issue time | learned vector; the encoder always has something to attend to when inputs are empty | built |
| GloTEC | sampled 2.5° cell (all qf>0 + ¼ of qf=0), lags 0/−1/−6/−24 h, ≤1200 tokens | geometry · Δt · anomaly of GloTEC foF2 vs IRI · quality flag (one-hot 0–5) · log TEC; own projection + kind embedding | built (v3) |
| spot activity (HF only: 1.8–28 MHz codes; paths 1000–3000 km so single-hop F2 dominates and line-of-sight / groundwave / long multi-hop are excluded; VHF+ never aggregated) | 5° midpoint cell · hour | geometry · Δt · per-band log-count anomaly (10 bands) · per-band SNR anomaly · log total spots · source id (WSPR / FT8, shared encoder) | 4b |

Local time is in the token so the encoder can represent "yesterday at this local time" (the 24 h
recurrence the Phase 3 lag analysis found: anomaly R² 0.28 at 24 h vs 0.07 at 12 h). Geometry is
geographic on purpose: magnetic coordinates are a learnable function of lat/lon, and a small
network learns them from 15k issue times; adding modip explicitly is a cheap later experiment.

Positional encoding is deliberately plain (lat/90, sin/cos lon, sin/cos local time through a
2-layer MLP): the tuned kernel says the anomaly's correlation length is ~2000 km, so attention only
needs coarse affinities, and linear coordinates plus an MLP represent that. Raw coordinates are
poor at sharp structure (EIA crests, terminator), so the first ablation if holdout residuals show
latitude structure is a small set of Fourier features of lat and local time. Year is *not* an
input on purpose: IRI/IGRF (Apex coefficients per year) carry secular drift, the train span is five
years, and a year feature mostly lets the model memorise per-year network changes (NOAA feed).
Solar-cycle position enters through F10.7 in the global token.

Token count: ≤2048 ionosonde rows per sample (random subsample when more). GloTEC adds ~200
sampled cells × 4 lags; spots add ~300 active cells. Budget ≈ 3.5k tokens.

## 4. Queries and targets

A query is a point `(lat, lon, t)` with lead `t − T ∈ (0, 24 h]`, carrying its own geometry,
local time, lead/24, IRI values at that point, and a kind flag (ionosonde / RO). Queries do not
see each other: the decoder is cross-attention only (`CrossAttnLayer`), so a query's answer depends
only on the encoded observations and maps are independent of how the grid is chunked. (v0 used
`nn.TransformerDecoderLayer`, whose query self-attention produced latitude banding on maps.)

Training targets are every observation in (T, T+24h]: ionosondes at all stations, RO profiles
anywhere. 20% of station **clusters** (co-located URSI codes merged) are withheld from the tokens
at each issue time; their future rows are the *holdout* targets and weigh ×2 in the loss, because
they measure spatial skill, which is the primary metric. RO rows are always holdout-like (never
inputs). Map production is the same forward pass with the 361×181 grid at each lead hour as the
query set (65k queries × 25 leads, batched).

## 5. Network

```
tok_in   : Linear(F_tok→128) · GELU · Linear(128→128)        (one per token kind)
glob_in  : Linear(6→128) · GELU · Linear                      → 1 token
null     : learned (1×128)                                    → 1 token
encoder  : 4 × TransformerEncoderLayer(d=128, 4 heads, FFN 512, pre-LN, dropout 0.1)
           key-padding mask over the ragged token set
qry_in   : Linear(F_qry→128) · GELU · Linear
decoder  : 2 × CrossAttnLayer(d=128, 4 heads, FFN 512, pre-LN)   cross-attention to memory only
head     : LayerNorm · Linear · GELU · Linear(128→6)  = 3 means + 3 log-variances (clamped [−6, 4])
```

1.39M parameters at d=128. Cost per sample: encoder O(N²d) with N≈2–3.5k, decoder O(N·M·d) with
M up to 12k queries; the query set is chunked, so memory is bounded by N. The model is small on
purpose: the exit criterion is beating the production GP, not saturating the data. Width and
depth are the first things to grow once it does.

## 6. Loss and training

- **Loss**: masked Gaussian NLL, `½(logvar + (a − μ)²/exp(logvar))`, summed over the three
  variables where the target exists, holdout queries weighted ×2. The σ head is calibrated by the
  same loss; PIT/coverage is reported by the harness.
- **Dropout as data**: per sample, a random 50–100% of tokens are kept (station dropout; the
  network has been shrinking since 2024 and will keep changing); the global token is zeroed 15%
  of the time; in 4b each *source* is dropped whole with its own probability, so no source is
  load-bearing. "Dropped in training" and "absent at inference" share the same padding mask.
- **Optimiser**: AdamW 3e-4, wd 0.01, 500-step warm-up, cosine to zero, grad-clip 1, batch 8
  issue times. ~18k samples at 3-h issue times over 2019-10 → 2024-12; 20 epochs is a few hours on
  one GPU.
- **Validation**: held-out ionosonde foF2 RMSE in MHz, printed next to IRI's on the same rows.
  The base run currently validates on 2025 samples for convenience; a reported number switches
  validation to a 2024 slice so 2025 stays clean for the rolling evaluation.

## 7. Rolling fine-tuning = production updates

`train/rolling.py`: for month M, fine-tune the previous checkpoint on the trailing 6 months
(2000 steps, lr 5e-5), then score M out of sample and save both. Chained over 2025 this is the
rolling-origin evaluation; run monthly on new data it is the production update job (snapshot →
samples → fine-tune → score → promote if not worse). The frozen test window 2026-01 → 2026-06 is
excluded from every fine-tune until the final report.

## 8. Evaluation gates (unchanged from PLAN.md)

1. Primary: held-out foF2 RMSE by lead bucket on the harness, paired against IRI, the tuned
   anomaly kernel (+9% over IRI) and the production GP (+3–4% over the kernel). All three are on
   the same rows via `eval/report.py`.
2. σ calibration (coverage of ±1σ, ±2σ; PIT histogram).
3. Map coherence: anomaly-field power at scales finer than station spacing, decay of the anomaly
   far from all inputs, frame-to-frame change in the fixed-local-time frame, sign flips.

## 9. Known limits, and what to try in order

1. Queries are independent given the memory (by construction since v1): no explicit spatial smoothness prior on the map. If
   the coherence gate fails, add a smoothness penalty on grid queries during training, not
   post-hoc filtering.
2. IRI's own bias (+0.3–0.4 MHz high in quiet 2024; PyIRI ~+0.1 vs Fortran) is absorbed by the
   anomaly, which is fine, but it means the "anomaly" mixes climatology error with weather. A
   learned global offset token could separate them later.
3. hmF2 truth is inconsistent between sources (ionosonde autoscaling vs RO vs GloTEC); expect
   hmF2 skill to stay modest regardless of architecture.
4. Multi-hop / non-midpoint spot geometry is ignored in the cell aggregates; the shared spot
   encoder can only learn what the aggregate exposes. Sporadic-E openings (summer, 28 MHz and up,
   500–2000 km) are not filtered and will look like high-MUF anomalies; the per-cell/band/UT-hour
   monthly baseline absorbs the seasonal part only. If 28 MHz activity turns out to hurt, drop it
   or add an Es proxy (foEs is in the ionosonde table). 1.8/3.5 MHz carried no signal in Phase 3
   and can be dropped from the token to save features.
5. No storm physics: Kp/ap enter only as global scalars. If storm-time skill lags, give the
   global token a short history (last 8 × 3-h ap) rather than a bigger model.
