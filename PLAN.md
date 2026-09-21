# Next-generation forecast model: plan

Goal: a model that ingests ionosondes, GloTEC, spot data (and future sources), and emits
global foF2/hmF2/MUF(3000)/TEC maps out to 24h+, evaluated offline against history at the
speed of compute. Replaces `pred` + `assimilate` + `diffusion` in production once it beats them.

Governing rule: **nothing is built on top of a component that hasn't been measured.** Every
phase has an exit criterion that is a number from the eval harness.

---

## Phase 0 — Data snapshot (offline, portable)

Pull everything out of the live DB and third-party feeds into versioned parquet on a HF
dataset repo, so any machine (local, kc2g, scranton, HF jobs) can run without DB access.

| Table / feed | Source | Rows | Notes |
|---|---|---|---|
| `ionosonde.parquet` | `measurement` ⋈ `station` | 22M | all columns; keep `cs`, `source` |
| `station.parquet` | `station` | ~100s | lat/lon as float (currently text) |
| `ro.parquet` | `cosmic_eval` | 6M | `time, lat, lon, fof2_true, hmf2_true, source`, deduped to one row per observation. Those columns are straight from the profiles, so this is honest truth. Coverage followed the live experiment system (2024 was missing Jan, Apr, May, Nov, Dec), so the full series is being re-derived from cosmic.ucar.edu by `data/fetch_ro.py`: cosmic-2 from 2019-10-01 (~29 MB/day, ~3k profiles/day), planetiq from 2023-04-05 (many days absent), one request per 3 s, raw tarballs + per-day parquet under `/kass/forecast/ro`. Parser verified exact against the old rows on a sample day |
| `glotec/YYYY-MM-DD.nc` | NOAA archive | 2025-05→ | raw netCDF (~15 MB/day, ~7 GB total) at `/kass/forecast/glotec`, mirrored by `data/fetch_glotec.sh`; archive begins 2025-05-12; mirrored 479 days (6.1 GB) on 2026-09-02 |
| `essn.parquet` | `essn` | small | production eSSN fits per run and series (24h/6h), with fit error; as-of-T by construction, so baseline 2 replays exactly as it ran and the offline re-implementation can be checked against it |
| `indices.parquet` | GFZ Kp/ap, SILSO SSN, F10.7 | small | needed for IRI-at-issue-time and stratification; see note below |
| `spots/wspr/YYYY-MM.parquet` | wsprdaemon ClickHouse `wspr.rx` on wd20 (10.112.0.20) | 12.5B rows, 2008→ (bulk 2019→), ~160 GB parquet | mirrored monthly by `data/fetch_wspr.sh` at ≤100 Mbit/s into `/kass/forecast/spots/wspr` (full table including `id`, so the DB can be re-seeded exactly; `time` is UTC). Hourly aggregates on HF (see Phase 3) |
| `spots/pskreporter/YYYY-MM-DD.parquet` | pskreporter ClickHouse `pskreporter.rx` on wd10 (10.112.0.10, primary serving replica: describe + stream only, no heavy queries) | 15B rows, 2024-12→, ~165 GB on disk | mirrored daily by `data/fetch_pskreporter.sh`, chained to start after the WSPR mirror finishes; full rows (no id column); has `mode` (FT8/FT4/…) |

**Indices as-of-T.** IRI wants the 13-month centred smoothed SSN (and IG12), which for the current
month needs 6 months of future data. The CHAIN `ig_rz.dat` files fill that in with a projection,
overwritten in place daily, so what production IRI actually saw on any past date is lost. The true
smoothed value is reconstructable for any date more than 6 months back, but using it in the harness
would give IRI (and everything built on it) a driver that production never has — the error in the
projection at solar max is tens of SSN units, which is a visible foF2 bias.

Plan: define our own estimator `ssn_hat(T)` computed only from SILSO monthly means through month
T−1 (plus a fixed extrapolation rule, e.g. hold the last 13-month mean or a McNish–Lincoln-style
fit), and F10.7/Ap from GFZ daily values ≤ T. Use it everywhere: harness baselines, model
training, and in production going forward, replacing the CHAIN projection so live and replay
inputs match exactly. Fidelity to what CHAIN said is not the goal; consistency between train,
eval and production is. Two checks before committing: (a) historical `ssn_hat(T)` vs true
smoothed SSN, and the implied IRI foF2 difference, to know the size of the driver error we're
carrying; (b) PyIRI vs Fortran IRI on a sample of points, since PyIRI takes indices as arguments
and skips the `ig_rz`/`apf107` file work. The anomaly model sees raw recent daily SSN/F10.7/Ap
as inputs too, so it can learn to correct residual driver error itself. `irimap`'s eSSN fit
becomes a baseline variant, not part of "IRI".

**Mechanics.** The DB port isn't exposed on the host, so the dump runs as a one-shot
`podman run --pod prop` of the existing `postgres:16` image with the 20 TB NFS volume
bind-mounted, streaming `COPY ... TO STDOUT | gzip` per table. The NFS is local disk on Andrew's
machine, so CSV→parquet conversion runs there (duckdb). Parquet is then pushed to the private HF dataset `arodland/iono-forecast`
(`snapshot/<date>/*.parquet`, `glotec/*.nc`) for scranton and rental GPUs. The kc2g server's `/home/prop/iri-index/predicted` holds only the latest index file, not an
archive, so as-of-T indices must be reconstructed (see the note above). Cheap fix going forward:
have the index fetch job keep a dated copy of each download so the archive starts accumulating now.

Two things to fix in the live pipeline now so future data is better than past data:

- Add `inserted_at timestamptz default now()` to `measurement`. We have no arrival times, so
  historical replay must assume a fixed latency (proposal: obs usable if `time <= T - 15 min`).
  Going forward we get the real number and can validate that assumption.
- Store raw RO profiles (time/lat/lon/foF2/hmF2/source), not just `cosmic_eval` comparison rows.

**Snapshot 2026-09-02 findings** (full card at `/kass/forecast/snapshot/2026-09-02/DATA_CARD.md`):
the NOAA feed stopped 2024-11-18 (known, not recoverable; 30 stations lost; 65 → 38 active
stations 2023 → 2025, TEC availability 0.79 → 0.03). Declining ionosonde coverage is a primary
motivation for the multi-source model. Test-period inputs are much sparser than training — the harness must
report skill vs. input-station count, and training should include station-dropout so the model is
used to it. Nine sites have 2–3 co-located URSI codes; holdouts are by location cluster (~50 km).
A few thousand cs-passing rows are physically impossible (foF2 > 25 MHz, hmF2 > 600 km, negative
TEC); the loader applies fixed physical bounds. 37% of rows have `cs = -1`.

Exit: parquet on HF, a one-page data card (row counts by year/source, quality-flag distributions,
station uptime), and a loader module used by everything downstream.

## Phase 1 — Evaluation harness ("replay")

The core abstraction: `forecast(issue_time T, model) -> callable(lat, lon, t) -> {fof2, hmf2, mufd, tec, σ}`.
Inputs available to the model are exactly the snapshot rows with `time <= T - latency`
(ionosonde, GloTEC, spots, indices *as published at T*). RO is target-only: it arrives too late
to be an input, which matches production.

**Targets and matching.** Ionosonde rows and RO rows with `time ∈ (T, T+24h]`, quality filter
`cs >= 75 OR cs = -1` (fixed once, never tuned). Match by station identity for ionosondes,
by exact lat/lon for RO. No spline interpolation of maps in the harness — models expose a point
query so they're all scored the same way.

**Holdout stations.** For nowcast/short-lead skill, evaluate on stations withheld from the
input set (rotate a random 20% of *location clusters* per issue time, seeded; co-located codes
are held out together). For pure forecast skill (lead ≥ 6h),
also score with all stations as input. Report both; they answer different questions.

**Metrics.** Primary (decided): foF2 RMSE on ionosonde holdouts, averaged over lead buckets
{0–1, 1–3, 3–6, 6–12, 12–24}h. Secondaries: hmF2, MUF(3000) RMSE; RO foF2/hmF2 RMSE;
log-accuracy (the `cosmic-vis` `stats-bootstrap.py` convention) for comparison with older
IRTAM-style studies. Everything also reported as
skill vs IRI: `1 - RMSE_model / RMSE_IRI`. Stratify by lead, day/night, |lat| band, Kp class,
station. Uncertainty calibration (PIT histogram / coverage of σ) once models emit σ.

**Map coherence (acceptance gate, not a score to optimize).** Point metrics can't see a map that
is right at the stations and garbage between them. Every model that produces maps also gets,
on the full 361×181 grid, at each issue time:

- *Spatial*: power spectrum of the anomaly field (model − IRI) by spherical-harmonic degree or
  zonal wavenumber. Power at scales finer than the local station spacing must not exceed that of
  the production GP maps by more than a fixed factor (set from the val period). Also the
  anomaly's gradient magnitude in observation-free regions (>1500 km from any input) must decay
  toward zero rather than hold structure.
- *Temporal*: for consecutive hourly frames of one forecast set, RMS frame-to-frame change of the
  anomaly, in both the geographic frame and the fixed-local-time frame (shift 15°/h). The
  local-time-frame change should be the smaller of the two (the anomaly should mostly co-rotate
  with the sun), and neither may exceed the observed hour-to-hour change at stations. Pixel-wise
  sign flips of the anomaly between consecutive frames, counted, as the flicker detector.

These are reported alongside the skill tables; a model that improves RMSE but breaks a coherence
bound is rejected. The bounds are calibrated on IRI + eSSN and the GP maps, which we know look
physically reasonable. Predicting the anomaly on top of IRI already gives a smooth mean field;
this gate is what keeps the learned part honest.

**Significance.** Paired block bootstrap over issue *days* (not samples — obs within a day are
correlated; this is what makes the current experiment system need months). Two models are
"different" only if the 95% CI on the paired skill difference excludes zero.

**Splits (temporal, frozen, written down here):**

| Split | Period | Use |
|---|---|---|
| train | 2019-10 → 2024-12 | model fitting (revised 2026-09-04; 2024 was val for Phases 1–3) |
| rolling eval | 2025-01 → 2025-12 | fine-tune to month M, score M+1, slide monthly; model selection |
| test | 2026-01 → 2026-06 | touched only for final reported numbers |
| live | 2026-07 → | the existing experiment system, as final confirmation only |

48h gap between splits. Note the shift: train is solar minimum + rise, val/test is maximum.
That is honest — it is the deployment condition — but it means we should also report val/test
skill for a train set that includes 2024 once the architecture is chosen.

Issue times: every 6h across the period (~2200 in test), so a full eval is thousands of forward
passes, not millions. Output is one parquet:
`(issue_time, lead, target_kind, target_id, lat, lon, var, truth, pred, sigma, model)`.
Reports are cheap scripts over that file. Port the plotting/bootstrap bits from
`~/software/cosmic-vis`, drop the rest.

Exit: the harness reproduces the known ordering **GP > IRI > current DiT** on the test period,
with CIs. If it doesn't, the harness is wrong (or the live finding was), and we stop until we know which.

**Eyeball tool (built in this phase, kept for every later one).** `forecast.py --at <time|now> --model <name>`
renders the 24h map set (all four variables, plus σ and anomaly-from-IRI panels) as PNGs or an
animated GIF, with input observations overplotted. `--at now` pulls live conditions from the
prop.kc2g.com APIs and today's GloTEC file; `--at <historical>` reads the snapshot through the same
point-in-time loader the harness uses, so what you look at is exactly what got scored. Every
baseline and model plugs into it the moment it exists. This is the successor to
`diffusion/app/live_forecast.py` and should land with the first baseline, not after the first model.

## Phase 2 — Baselines (the skill ladder)

All cheap, all in the harness before any new ML:

1. **IRI-2020** with indices as-of-T (Phase 0 note).
2. **IRI + eSSN**: what `irimap` does today. Separates "better index" skill from "assimilation" skill.
3. **Persistence**: last observation per station, held flat.
4. **IRI + persisted anomaly**: `pred = IRI(t) + (obs(T) - IRI(T)) * exp(-(t-T)/τ)` per station,
   spatially spread to holdout/RO points with a simple distance kernel. τ and kernel scale tuned
   on val. This is a 50-line model and I expect it to be surprisingly hard to beat at 1–6h.
5. **Production GP** (`pred` + `assimilate`) replayed offline. Needs its inputs reconstructed
   point-in-time (eSSN too). If replay is too painful, use its live `pred_eval`/`cosmic_eval` rows
   for the overlapping period as a fixed reference instead.

IRTAM is not a baseline: it's 3 days delayed so it can't be an input, and recent numbers have it
below IRI even at 0–4h. The existing `fof2_irtam` columns stay in the snapshot for historical
comparison only.

Exit: a table of skill-vs-IRI by lead for 1–5 on val and test. This table is the bar every
subsequent model must clear, and it tells us where skill lives (short lead? night? low lat?).

## Phase 3 — Source value checks (before feeding anything to a model)

Each candidate input gets a cheap, model-free test of whether it carries information about
the *anomaly* (obs − IRI), which is the thing to be predicted:

- **GloTEC**: at ionosonde locations, correlate GloTEC foF2/hmF2/TEC anomaly with ionosonde
  anomaly at the same hour; by region and quality flag. Also: does GloTEC anomaly at T predict
  ionosonde anomaly at T+1..6h beyond what the ionosonde's own anomaly at T does?
- **Spots**: define an hourly, per-path-midpoint-cell proxy (e.g. highest band with ≥k spots on
  1000–3000 km paths ≈ MUF proxy; spot count by band as a coverage/absorption proxy). Correlate
  with ionosonde MUF(3000) anomaly nearby. This also fixes the aggregation format we'd ingest.
- **Indices**: Kp/ap at T vs. anomaly at T+h — how much of the storm signal is already in the
  indices?

Exit: for each source, a partial-correlation / incremental-R² number on val. A source that shows
nothing here does not go into the model; one that does goes in with a known ceiling to compare
model gains against.

## Phase 4 — Model, in order of increasing ambition

Design stance (agreed): **predict the anomaly field, supervise on real observations,
deterministic with uncertainty first.** Hard requirements: 361×181 map output, per-point σ,
cheaper to train than the DiT. Diffusion's appeal was learning spatial structure; here the spatial
structure of the mean comes from IRI plus a learned anomaly field, and the time dimension is
handled by lead-time conditioning rather than by generating trajectories.

Why not restart with diffusion on IRI maps: the previous model's training signal was synthetic
IRI maps with noised pixel samples as fake observations, so its learned prior *is* IRI, and real
observations only entered through CFG-style nudging and a late fine-tune. The 22M real
observations were barely used as supervision. Meanwhile the metric we care about is point error,
for which a generative model's best output is its conditional mean anyway.

**4a. Anomaly regressor (the new core).** A set-to-function model:
- Input tokens: observations in the last 24–48h — `(lat, lon, Δt, kind, values, weights)` — plus
  indices and IRI context at the token location. Same 11-ish column layout as before is fine.
- Query: `(lat, lon, t, IRI(lat,lon,t))` for any point, including the whole 181×361 grid.
- Output: mean and log-variance of anomaly per variable. Loss: Gaussian NLL.
- Training: real data only. Sample an issue time, mask ~20% of stations as targets plus all
  future obs in (T, T+24h]; the model sees the rest. This is exactly the harness's holdout
  protocol, so train and eval objectives coincide.
- Architecture: a transformer encoder over obs tokens with cross-attention from query tokens
  (a conditional neural process). Small first (~10–30M params); it must beat baseline 3 on val
  before it gets bigger. Locality (the `LOCAL_OBS_ATTENTION_DESIGN.md` bucketing) becomes a
  measured optimization later, not a prerequisite.

Exit: beats baseline 4 and the production GP on val primary metric with CI excluding zero,
calibrated σ, and passes the coherence gate. If the gate fails, the fix is in training (spatial
smoothness / temporal consistency penalties on the query grid, or fixed-local-time coordinates
in the query encoding), not in post-hoc filtering of maps.

**4b. Add sources.** One at a time, each as a new token kind (GloTEC pixels subsampled, spot
aggregates as cell tokens). Train with source-level dropout so no source is indispensable.
Each source must move the val number; if it doesn't, it's out, regardless of Phase 3.

**4c. Generative upgrade (only if needed).** If ITURHFPROP or the site needs realistic
small-scale structure or true ensembles, add flow matching *on the anomaly residual* conditioned on
the 4a encoder. Evaluate with CRPS / spread-skill, and confirm point RMSE doesn't regress.
The DiT/VAE/CFG machinery is not carried over unless this phase shows it's needed.

## Phase 5 — Production

- New service (one directory, one `uv` project, one container) exposing the same map/point API
  the site and ITURHFPROP consume today; `pred`/`assimilate`/`diffusion` stay until the live
  experiment system confirms the offline result over ~1 month, then are retired.
- Model artifacts and eval parquet on HF; training on scranton (L40), dev on local ROCm,
  inference on kc2g (A4500).
- Live eval continues to write to the same parquet schema as the harness so offline and online
  numbers are directly comparable.
- **Continual fine-tuning is the rolling-origin step, productionised** (Andrew, 2026-09-04): one
  `update` job = snapshot the new month (DB dump → parquet, mirrors topped up) → build samples for it
  → fine-tune from the current checkpoint with a fixed small budget and low LR → score the new month
  out of sample (it is the rolling-eval month) → promote only if not worse than the incumbent on the
  primary metric → push checkpoint + eval parquet to HF. The same script does the offline rolling
  evaluation over 2025, so the production procedure is validated before it runs live. The frozen
  test window is never included in any fine-tune until the final report.

---

## Later: long-distance paths (after Phase 4a is proven)

Goal (Andrew, 2026-09-04): predict trans-Atlantic / trans-Pacific paths (4000 km+), and use spots
on those exact paths for short horizons. Attribution to a map cell is the hard part; two routes:

1. **Path-level model, no attribution.** Make the *path* the unit: token = (tx cell, rx cell,
   band, hour, activity anomaly, SNR anomaly); query = (tx, rx, band, t). Target = openness (any
   spot on that path/band in hour t), so the truth is the spot archive itself and the metric is
   log-loss / AUC on future spots. Persistence on the same path (1 h, 24 h earlier) is the baseline
   and probably strong at short horizon. This is the "where will I be heard" product directly.
2. **Control-point attribution for the map model.** Long spots enter the map model as weighted
   tokens at ITU-style control points (P.533: 1000 km from each end plus the midpoint for
   >7000 km; hop points from `raytrace/` for the general case), each carrying the band/activity
   anomaly and a hop-count feature. The encoder learns how much to trust each point.
   Evaluate by whether adding long-path tokens improves ionosonde/RO skill at the control points.

Both can be tested cheaply once the harness accepts path queries: add `kind=path` targets built
from hourly path/band openness in the aggregate, and let the map model's MUF along control points
be the "physics" baseline for openness. Do (1) first; it needs no attribution and answers the
product question.

## Repo layout (new top-level dir; name TBD)

```
<name>/
  pyproject.toml          one uv project
  data/     snapshot.py   DB/feeds -> parquet on HF; loaders
  eval/     replay.py     issue-time loop, matching, parquet out
            report.py     skill tables, bootstrap CIs, plots (from cosmic-vis)
  baselines/              iri.py essn.py persistence.py anomaly_decay.py gp_replay.py
  models/                 4a onward
  forecast.py             eyeball tool: live or historical conditions -> map set PNG/GIF
  train.py
```

`diffusion/` becomes read-only reference. Nothing is imported from it; specific pieces
(IRI wrapper, TinyIonoVAE if 4c happens) are copied over once they've been re-verified.

## Known defects found while surveying (fix in Phase 1)

- `holdout-eval/heval.py:132` ties are broken by `hours_ahead`, not `time_diff`; the matched
  measurement is arbitrary within a bucket.
- `diffusion/app/live_forecast.py` and `server.py` call `history` at `/all_stations.json`, which
  doesn't exist in `history/app/main.pl` — deployed code differs from repo.
- `station.latitude/longitude` are text columns.
- Top-level `CLAUDE.md` says `assimilate` consumes GloTEC; it doesn't (only `diffusion/app/glotec.py` does).

## Decisions log

- 2026-09-02: primary metric foF2 RMSE; log-accuracy secondary. Deterministic-first accepted.
  IRTAM dropped as baseline. Spot archive fetch (1–2 TB) is in Phase 0. RO truth from
  `cosmic_eval` is acceptable. Indices must be reconstructed as-of-T; PyIRI preferred, pending
  validation against Fortran IRI.

- 2026-09-02: **IRI engine = PyIRI, driver = trailing 81-day mean observed F10.7** (`data/drivers.py`).
  On 20k RO points 2023–2026 (`analysis/compare_pyiri_fortran.py`), PyIRI+f107_81 vs the Fortran
  IRI production ran with CHAIN indices: foF2 bias +0.07, RMSE 0.47 MHz; skill vs RO truth identical
  (1.62 vs 1.63 MHz). hmF2 differs more (bias −10 km, RMSE 29 km) but PyIRI is closer to truth
  (51.9 vs 56.2 km). The 13-month-SSN driver was worse (foF2 bias +0.49). Full grid × 24 h in ~9 s.
- 2026-09-02: HF dataset `arodland/iono-forecast` holds snapshot parquet + GloTEC archive.

- 2026-09-02: replay harness (`eval/replay.py`, `eval/report.py`) and baselines iri / persistence /
  anomaly_decay (`baselines/simple.py`) run end to end on 2024 issue times. Output schema as specified
  above. Smoke numbers (4 issue times, not significant): IRI foF2 RMSE ≈1.0–1.5 MHz at ionosondes,
  1.6 MHz at RO; anomaly_decay (τ=6h, L=1000 km, untuned) a few % better than IRI at ionosondes,
  slightly worse at RO.

- 2026-09-02: **Phase 2 baselines on val 2024** (1464 issue times, 6 h step; `/kass/forecast/eval/val2024/REPORT.md`).
  foF2 RMSE, ionosonde holdout clusters: IRI 1.37–1.39 MHz flat across leads; IRI+anomaly_decay
  (τ=6 h, L=1000 km, untuned) 1.17 / 1.20 / 1.25 / 1.32 / 1.37 for 0–1 / 1–3 / 3–6 / 6–12 / 12–24 h,
  skill +4.6% [+4.1, +5.3]. Full mode (station was an input): 0.88 MHz at 0–1 h, skill +6.8%.
  Persistence beats IRI only inside 1 h. At RO points (low-lat, far from stations) anomaly spreading
  is slightly *worse* than IRI (−0.5%, hmF2 −0.8%): the Nadaraya–Watson kernel extrapolates full
  anomaly magnitude to any distance; needs shrinkage toward zero with distance (tune λ, τ, L on val).
  Q4 alone (solar max) shows larger IRI error (1.6 MHz) and larger anomaly skill (+8%).

- 2026-09-02: **anomaly_decay tuned on val 2024** (293 issue times, 36-point grid; `eval/rank.py` on
  `val2024/tune_anomaly.parquet`). Best: τ=48 h, L=2000 km, shrink=1.0 → holdout foF2 RMSE
  1.11 / 1.18 / 1.21 / 1.24 / 1.26 by lead bucket vs IRI 1.36–1.39, primary 1.20 vs 1.38 (**skill
  +12.8%**, up from +4.6% untuned). Shrinkage fixes the RO regression (1.73 vs IRI 1.74). Anomalies
  persist well past 24 h: even at 12–24 h lead the tuned model keeps ~8% skill, so the "decays to
  IRI by 12 h" picture was an artifact of τ=6. Optimum is on the grid edge in all three
  parameters; the second pass (τ 48–∞, L 2000–5000 km, shrink 1–3) is a plateau: all 18 variants
  within 0.01 MHz. Keep τ=48 h, L=2000 km, shrink=1. This family is exhausted; further gains need a
  learned (anisotropic, local-time-aware) kernel. This is the bar for Phase 4a.
  Harness note: holdout seed was Python's salted str hash (different per process) until fixed on
  2026-09-02; runs before the fix are not row-paired with each other across files.

- 2026-09-02: **eSSN driver mapping.** `irimap.F90` passes eSSN as R12 and derives IG12 and F10.7
  with its own quadratics; PyIRI takes F10.7 only and derives IG12 internally. Feeding PyIRI the
  essn table's `sfi` column underestimates IG12 by ~10 (foF2 low) — the first IRI+eSSN replay showed
  no gain and −10% MUF for that reason. Correct driver is `R12_2_F107(essn)`, which reproduces the
  Fortran IG12 exactly (`data/drivers.py::f107_for_essn`). Production reference (2023 control
  branches, 3697 hourly holdouts): IRI 1.238, IRI+eSSN 1.169 (+5.6%), GP 1.087 (+12.2%) MHz foF2.

- 2026-09-03: **IRI+eSSN on val 2024, corrected driver** (1464 issue times): holdout foF2 primary 1.335 vs
  IRI 1.389, skill +3.1% [−0.3, +5.5] (production showed +5.6% in 2023). But eSSN makes RO (low-lat)
  foF2 *worse* (1.93 vs 1.75 MHz) and MUF(3000) worse (4.28 vs 4.09) — a single global scalar fitted
  to a mid-latitude-heavy network over-corrects the equatorial ionosphere and the M3000 response.
  eSSN+anomaly: +7.5% at holdouts, RO still worse than IRI (1.88). The "global" term needs at least
  latitude structure; that's a design input for the learned model.

- 2026-09-03: **tuned anomaly kernel vs production GP, paired on production's 2023 holdouts** (79k
  common rows, foF2): GP −4.1% better than the kernel [−5.6, −2.7]; kernel +1.8% over production
  IRI+eSSN; GP +9.8% over production IRI; kernel +6.1% over production IRI. By lead the GP leads by
  0.04–0.06 MHz everywhere. So 50 lines recover ~60% of the GP's skill over IRI, and the GP is the
  first thing the learned model must beat. Caveat: our PyIRI IRI (trailing-81-day F10.7) is 3.6%
  worse on foF2 than production's Fortran IRI (CHAIN projected indices) in 2023, a rising-activity
  year — but the F10.7 window is not the cause (27/81/365-day all match the Fortran to 0.23 MHz at
  RO points). At stations PyIRI runs +0.11 MHz above the Fortran, mostly below 50° latitude; both use
  URSI coefficients, so the likely cause is the Fortran's measured IG12 from the CHAIN file vs
  PyIRI's IG12 derived from F10.7. Unresolvable without archived index files (now being kept).
  Treated as a known level bias that the anomaly layer absorbs at stations.
  jf-flag audit (2026-09-03): PyIRI ≡ Fortran for foF2 coefficients (URSI, jf5=f) and no foF2 storm
  model (jf26=f in both drivers). Differences: **hmF2** — Fortran point driver uses AMTB-2013
  (jf39=f, jf40=t), `irimap` uses Shubin-2015 (jf40=f), PyIRI only has the old BSE-1979 M(3000)
  formula; that is the 28 km RMSD / −10 km bias. **Indices** — Fortran reads measured IG12 + daily
  and 81-day F10.7 from files; PyIRI derives everything from one F10.7. B0/B1: Fortran ABT-2009,
  PyIRI NeQuick-style thickness (irrelevant to foF2/hmF2/MUF). Topside/TEC not comparable.

- 2026-09-03: **Phase 2 ladder, val 2024, consolidated pass** (1464 issue times, fixed seed, all
  paired; `val2024/REPORT_all.md`). foF2 primary / RO RMSE (MHz) and paired skill vs IRI at holdouts:

  | model | primary | RO | skill (holdout) | skill (RO) |
  |---|---|---|---|---|
  | IRI (PyIRI, f107_81) | 1.389 | 1.745 | — | — |
  | IRI + eSSN | 1.335 | 1.933 | +3.1% [−0.3, +5.5] | −10.8% |
  | IRI + eSSN + anomaly | 1.242 | 1.883 | +7.5% | −7.9% |
  | **IRI + anomaly (τ48, L2000, s1)** | **1.230** | **1.730** | **+8.8% [+7.7, +10.1]** | **+0.9%** |
  | persistence (full mode only) | — | — | far below IRI beyond 1 h | — |

  The anomaly kernel on plain IRI beats every eSSN variant on every metric; eSSN adds nothing once
  a spatial anomaly field exists and hurts low latitudes. Phase 2 exit criterion met, pending the
  corrected eSSN models on the 2023 production holdouts (queued).

- 2026-09-03: **2023 production holdouts, complete and paired** (79k rows, foF2 skill vs production IRI):
  GP +9.8%, eSSN+anomaly +7.2%, anomaly +6.1%, our IRI+eSSN +4.2% ≈ production IRI+eSSN +4.4% (so
  the PyIRI eSSN reproduction is validated), our IRI −3.6%. Versus the GP: eSSN+anomaly −2.9%
  [−3.7, −2.0], anomaly −4.1%. In 2023 eSSN still adds ~1% on top of the kernel at mid-latitude
  stations; in 2024 it added nothing and hurt RO. **Phase 2 exit: met.** Bar for Phase 4a =
  production GP (~3–4% over the best 50-line baseline, at every lead, on foF2, MUF and hmF2).

- 2026-09-03: **PyIRI entry point = `sh_library.IRI_density_1day`, hmF2 = SHU2015.** The legacy
  `main_library` call I'd used is BSE-1979-only; the SH path exposes `hmF2_model`. On 8k RO points
  2023–26: foF2 unaffected by the choice (bias +0.08, RMSE 0.49 vs Fortran; 1.64 vs truth ≈ Fortran
  1.65). hmF2 vs RO truth: SHU2015 48.8 km, AMTB2013 54.0, Fortran point driver (AMTB) 55.4,
  BSE1979 76.7. AMTB reproduces the Fortran point driver to 11.5 km RMSD, confirming the flag
  audit. SHU2015 also matches what `irimap` serves. Expect a −16 km offset vs `pred_eval` hmF2.
  Cost: the SH path recomputes Apex coordinates per UT step (~10× slower), so IRI now runs through
  a per-day map cache (`baselines/iri_cache.py`: 1° hourly global maps per quantised F10.7 level,
  linear in F10.7, trilinear to targets) shared by every model and the eyeball tool. Cache vs
  continuous IRI: foF2 0.015 MHz RMS (the old direct path's 15-min time rounding was 0.05). One
  map = 18 s after patching (was 38 s); warm-cache replay ≈ 1 s per issue time. Cache lives at
  `/kass/forecast/iri_cache`. `baselines/pyiri_patch.py` monkeypatches PyIRI 0.1.7 (pinned exactly):
  memoised `Apex_geo_qd` per (points, year) and a vectorised `real_SH_func` (one
  `assoc_legendre_p_all` call + cached normalisation table); both bit-exact vs the originals.
  Upstream candidates. What remains is inherent: the SH basis is evaluated in the MLT frame, so it
  is rebuilt for every UT step (N_T × N_G × 900 terms).

- 2026-09-04: **Ladder regenerated on the sh_library/SHU2015 engine via the map cache** (`val2024_v2/`):
  identical within noise to the legacy-engine run — IRI 1.396 (was 1.389), anomaly 1.233 (1.230),
  skill +9.0% [+7.8, +10.2] (was +8.8%); RO +1.0%; eSSN variants unchanged in ordering. hmF2 at
  stations also unchanged (36.3 km). The engine switch is validated; `val2024_v2` is now the
  reference set and the earlier `val2024/` files are superseded.

- 2026-09-04: **2023 production-holdout comparison regenerated on the new engine** (`gp2023_v2/REPORT.md`):
  vs GP, foF2: anomaly −3.8% [−5.2, −2.5], eSSN+anomaly −2.6% [−3.4, −1.8]; our IRI vs production
  IRI −3.2%. hmF2 vs GP with SHU2015: anomaly −1.6%, i.e. the kernel is within 2% of the GP on hmF2
  and 3–4% on foF2/MUF. All within noise of the legacy-engine run. **Phase 2 closed.**

- 2026-09-04: **Phase 0 mirrors complete.** RO re-derived from UCAR: `ro_full.parquet`, 9.0M obs,
  continuous 2019-10 → 2026-09 (on HF; loader now uses it). WSPR 193 GB (222 months), pskreporter
  186 GB (641 days), both on `/kass/forecast/spots` only (too large for HF; aggregates go there).
  GloTEC sampled at station cells (`analysis/glotec_at_stations.py` → `eval/glotec_stations.parquet`,
  7.5M rows, 65% with qf>0 at station cells). Phase 3 GloTEC check underway.

- 2026-09-04: **GloTEC nowcast agreement at station cells** (2025-05 → 2026-09, 2.4M matched rows):
  GloTEC foF2 (from NmF2) vs ionosonde: RMSE 0.88 MHz at qf=5 (bias +0.09, r=0.94), 0.96–1.08 at
  lower flags, *including qf=0*. IRI is ~1.4 at these stations. hmF2 RMSE ~39 km (bias +20).
  Ionosonde TEC is not the same quantity (bias +11 TECU, r 0.76–0.84). **At RO points** (1.57M,
  80% |lat|<30, `analysis/glotec_at_ro.py`): foF2 RMSE 1.27 MHz at qf=5, 1.32–1.36 at qf 1–4,
  1.53 at qf=0 (r 0.88–0.92; bias −0.1); by band at qf≥3: 0.7–0.8 poleward of 50°, 1.0–1.3 at
  mid/low, 1.5 at |lat|<15 night. IRI at RO was 1.75 in 2024. Full value check (anomaly
  correlation vs IRI, lag/incremental R²) in `analysis/glotec_value.py`, pending the 2025–26 IRI cache.

- 2026-09-04: **Phase 3, GloTEC: in.** (`analysis/glotec_value.py` → `eval/GLOTEC_VALUE.md`, 2025-05 → 2026-09.)
  Nowcast foF2 RMSE, GloTEC vs IRI: stations 0.93 vs 1.38 MHz at qf=5 (−33%), 1.01 vs 1.41 even at
  qf=0; RO points 1.27 vs 1.60 at qf=5 (−21%), 1.53 vs 1.62 at qf=0. hmF2 from GloTEC is worse than
  IRI everywhere (ignore it). Predictive value at a station that *has* an ionosonde: GloTEC adds
  ~nothing beyond the station's own anomaly (incremental R² ≤0.007), as expected; alone it carries
  about half the information of a co-located ionosonde (R² 0.34 vs 0.65 at 1 h). Its value is
  spatial — a dense nowcast where no station exists — so it enters the model as a map-like source,
  and the honest test of its forecast value is the harness with GloTEC as an input to the anomaly
  field (kernel + GloTEC anomaly, Phase 4b). Side finding: station-anomaly R² at 24 h (0.28) is far
  above 12 h (0.07) — strong same-local-time recurrence that an exponential decay cannot represent;
  the learned model must see "yesterday at this local time".

- 2026-09-04: **WSPR profile** (June 2025: 162M spots, 5k receivers, 112k transmitters): bands are
  integer-MHz codes; HF traffic is 7 (47M), 14 (53M), 10 (28M), 3 (9M), 18/21 (7–8M), 28 (4M).
  71M spots/month on 1000–3000 km paths (single F2 hop). Lat/lon never (0,0); 0.2% zero distance.
  Phase 3 spot check: hourly × 5° path-midpoint cell × band counts/SNR (`analysis/wspr_aggregate.py`),
  then "highest band with ≥k spots" as a censored MUF proxy vs ionosonde MUF(3000) anomaly.

- 2026-09-04: **Phase 3, WSPR, first proxy: weak.** (`analysis/wspr_value.py` → `eval/WSPR_VALUE.md`, 2024,
  285k station-hours, 56 stations; a 5° midpoint cell contains a station 37% of the time.)
  "Highest HF band with ≥3 spots on 1000–3000 km paths" tracks climatology: mean MUF by proxy band
  ≈ IRI's mean; anomaly correlation 0.21–0.26; R² 0.04; no forecast value beyond the station's own
  anomaly. The proxy is censored by band occupancy, so this rules out *this formulation*, not spots.
  **Activity-anomaly formulation works, weakly**: per band, log-count minus that cell/band/UT-hour's
  monthly median correlates with the station MUF(3000) anomaly at 0.23–0.30 on 14/18/21/24/28 MHz,
  ~0.15 on 10 MHz, ~0 on 1.8/3.5 MHz (physically right: high bands open when MUF is high); SNR
  anomalies weaker (≤0.15). All bands together: leave-one-month-out R² 0.094 for the same-hour MUF
  anomaly. For scale, GloTEC alone explains 0.34 of the 1-h-ahead foF2 anomaly. **Verdict: WSPR is
  in as a secondary source**, formulated as per-band activity anomalies on midpoint cells; expect
  small gains, mostly where neither ionosondes nor GloTEC cover. pskreporter (FT8, ~5× denser) to
  be checked the same way.

- 2026-09-04: **Phase 3, pskreporter FT8: in, stronger than WSPR.** (`analysis/spot_activity_value.py`
  → `eval/PSK_VALUE.md`; 2025-05-12 → 2025-12, 47k station-hours, 22 stations.) Per-band activity
  anomaly vs station MUF(3000) anomaly: 0.27–0.38 on 18/21/24/28 MHz, 0.15 on 14, −0.16 on 7 MHz
  (band shifts up when MUF is high); SNR anomalies 0.21–0.25 on 14–21 MHz. All bands together:
  leave-one-month-out R² 0.18 (WSPR: 0.09). Highest-band proxy useless again (R² 0.01).
  Mirror gap: many pskreporter daily files are empty (Feb 2025 mostly, Dec 2025 nearly all, parts of
  Jan/Jun/Oct/Nov) — the export returned zero rows for those days, not a fetch failure. Whether the
  source table has those gaps needs checking on wd10 (heavy query; needs approval).

- 2026-09-04: **Phase 3, indices: in (small, storm-specific).** (`analysis/indices_value.py` → `eval/INDICES_VALUE.md`, 2024.)
  ap at T plus 24-h max ap adds R² +0.016 at 6 h and +0.026 at 12 h to the station's own anomaly,
  ~0 at 1 h and 24 h. The effect is the storm-time negative phase: mean foF2 anomaly vs IRI goes from
  −0.4 MHz (quiet) to −1.5 (mid-lat storm) and −1.8 (high-lat storm), with RMSE doubling. Also
  visible: IRI is biased high by 0.3–0.4 MHz even in quiet 2024 — the "global term" the eSSN chased.
  **Phase 3 closed.** Sources for Phase 4, in order of measured value: ionosondes ≫ GloTEC (dense
  nowcast) > FT8 activity anomalies > WSPR activity anomalies ≈ indices (storm phase).

- 2026-09-04: **Split revision for Phase 4 (agreed).** Train 2019-10 → 2024-12, evaluate on 2025+, and
  because GloTEC (2025-05→) and FT8 (2024-12→, with capture gaps that are real, not retrieval) barely
  exist in the train window: (1) one shared *spot-activity* token encoder for WSPR and FT8 (same
  representation, a source-id embedding), so the spot pathway is trained on WSPR's 2019+ history and
  FT8 needs only adaptation; (2) rolling-origin evaluation over 2025–2026 — fine-tune with all
  sources on data up to month M, score month M+1, slide monthly — so every month after the sources
  appear is scored out of sample while all data gets used; (3) 2026-01 → 2026-06 stays frozen as the
  final report window. Source-level dropout (whole source dropped per sample) plus station dropout
  in training, so no source is load-bearing; missing-at-inference and dropped-in-training share the
  same mask.

- 2026-09-04: **Phase 4a build started** (`train/`). Samples = the harness protocol frozen to disk, one
  `.npz` per issue time every 3 h: ≤2048 ionosonde tokens from [T−24h, T−15min] with 20% of clusters
  withheld; queries = every observation (ionosonde + RO) in (T, T+24h]; targets = anomaly vs cached
  IRI, normalised by (1.5 MHz, 40 km, 4.5 MHz). Token features: lat, sin/cos lon, sin/cos local time,
  Δt, IRI values, anomalies + presence flags, confidence. Query features: same geometry + lead + IRI +
  kind. Global token: F10.7 (81-day, daily), ap now, 24-h max ap, day-of-year. Model
  (`train/model.py`): CNP-style transformer — 4 encoder layers over tokens (+ a learned null token
  and the global token), 2 cross-attention decoder layers for queries, heads for mean and log-variance
  ×3; d=128, ~1–2M params. Loss: masked Gaussian NLL, held-out queries weighted ×2. Training-time
  dropout: random 50–100% of tokens kept, global token zeroed 15% of the time. Val metric = RMSE on
  held-out ionosonde foF2 (the primary metric). Ionosondes + indices only first; other sources after.
  Training runs on the GPU by Andrew (`uv run --extra train train/train.py ...`); sample building and
  a CPU smoke test here.

- 2026-09-04: architecture write-up in `forecast/MODEL.md` (tokens, network, loss, rolling fine-tune, gates, known limits).
- 2026-09-04: CPU smoke test of `train/train.py` on 4 samples passes end to end (1.39M params);
  full sample build (2019-10 → 2025-12, 3-h issue times, ~18k samples) chained behind the
  training-years IRI cache. Continual fine-tuning added to Phase 5 as the productionised
  rolling-origin step (same script as the offline rolling eval).

- 2026-09-04: **training samples built**: 15,235 train (2019-10-01 → 2024-12-31, every 3 h, 9.5 GB)
  and 2,920 rolling-eval (2025, 1.5 GB) at `/kass/forecast/samples/{train,eval2025}`; tarballs
  pushed to HF `samples/` for the Scranton L40. IRI cache now 8.6k maps covering 2019-10 → 2026-09.
  Next: base training on the GPU (Andrew), then `predict.py` + `report.py` against the ladder.

- 2026-09-05: **2025 ladder** (`eval/val2025/`, 2920 issue times at 3 h, same holdout seed as the
  samples): IRI primary 1.441 MHz (RO 1.713); tuned anomaly kernel 1.204 (+13.4% [+12.3, +14.5]),
  RO 1.658. First smoke training of the 4a model (A4500, 325 s/epoch): held-out foF2 1.025 vs IRI
  1.447 on the sample rows after 3 epochs, i.e. already well past the kernel; paired numbers pending
  `predict.py` output.

- 2026-09-05: **production GP maps are archived** at `/kass/prop-archive/prop-archive/<id//1e6>/<id//1e3%1e3>/<id>/{assimilated,irimap}/`
  (one run per 15 min from 2023-04, 25 hourly `.h5` files per run: `/maps/{fof2,hmf2,mufd,foe,md,gyf}`
  181×361 on lat −90..90 / lon −180..180, `/stdev/{fof2,hmf2}`, `/stationdata/pred` = stations
  assimilated). `eval/gp_maps.py` scores them at the harness targets into the replay schema. Fairness:
  production assimilates every usable station, so at those stations its rows are *full* mode; rows
  at stations it did not assimilate are labelled *holdout* (a small, selection-biased set); RO rows
  are fair for everyone. So the GP-vs-model comparison on 2025 is on RO points and full mode; the
  holdout comparison stays on the 2023 production-holdout set.

- 2026-09-05: **production GP on 2025** (`eval/val2025/gp_q*.parquet`, 2910 runs at 3 h; fair views only):
  full mode (all stations assimilated), foF2 paired skill vs IRI: GP +27.0% [+25.6, +28.5],
  kernel +16.6%, IRI+eSSN +12.5%; GP 0.73 MHz at 0–1 h rising to 1.07 at 12–24 h. **RO points**
  (nobody's input): GP −3.4% [−5.0, −1.7], IRI+eSSN −8.1%, kernel +3.2% — production is worse than
  climatology away from its stations in 2025, the eSSN low-latitude over-correction carried
  through. So the bar on 2025 is: match the GP in full mode (+27%) *and* beat IRI at RO points.
  The GP's non-assimilated-station rows ("holdout", 1.118) are selection-biased; not a comparison.

- 2026-09-05: **model_v0 (first 4a run, ionosondes + indices only, ~20 epochs on 2019-10 → 2024-12) on
  2025, paired** (`/kass/forecast/model_v0.parquet`; val was 2025 itself, so this is a smoke number).
  Holdout stations, foF2: primary 1.019 vs kernel 1.204 / IRI 1.441 → +17.8% over the kernel
  [+16.5, +19.0]; flat across leads (1.01 at 0–1 h to 1.06 at 12–24 h). Full mode vs the production
  GP on 9.5M common rows: foF2 +1.8% [+0.8, +3.0], MUF +2.7%, hmF2 ±0. RO points vs IRI: foF2
  +18.5% [+17.6, +19.3] (GP −3.4%), hmF2 +10.9% (GP −6.8%). σ calibration: 66.7% within ±1σ, 93.5%
  within ±2σ at holdouts; 70.3% / 94.8% at RO (ideal 68.3 / 95.4). **Phase 4a exit criterion met on
  the smoke run**: matches the GP where the GP is strong, beats everything where it is weak, with
  calibrated uncertainty. Caveats to close before reporting: validation on a 2024 slice instead of
  2025; the 2023 production-holdout comparison; the coherence gate on full maps.

- 2026-09-05: **eyeball tool** `forecast/forecast.py`: `--at <issue time> --model <baseline|ckpt.pt>`
  renders per-lead panels (forecast, anomaly vs IRI with input-station dots, σ for checkpoints) and a
  GIF, using the same token/query builders as training and the same IRI cache. Historical only;
  `--at now` via the prop.kc2g.com API (`/stations.json` + `/sonde_export`) next.

- 2026-09-05: **v0 maps show latitude banding: query self-attention in the decoder.** `nn.TransformerDecoderLayer`
  lets queries attend to each other, so a grid point's value depended on its 8192-query chunk
  (measured: chunk 8192 vs 1024 changes the foF2 anomaly by 0.06 MHz mean, 0.40 max, on a field of
  σ 0.32). Fixed in `train/model.py` with a cross-attention-only `CrossAttnLayer` (a proper CNP;
  output provably chunk-invariant); v0 checkpoints still load via `query_self_attn=True`. **v1 =
  retrain with the new decoder**; v0's scores stand but its maps do not pass the coherence gate.

- 2026-09-05: **coherence gate implemented** (`eval/coherence.py` on grids saved by `forecast.py`;
  `eval/gp_grids.py` exports an archived production run for reference). One issue time
  (2025-06-15 12:00, 9 leads), v1 model vs production GP vs kernel: anomaly RMS 1.00 / 1.28 / 0.44
  MHz; fine-scale (k>20) variance fraction ~0 for all; gradient far from stations 0.042 / 0.030 /
  0.010 MHz per 100 km (near: 0.042 / 0.026 / 0.006); mean |anomaly| >1500 km from stations 0.77 /
  1.14 / 0.30; RMS change per hour 0.19 / 0.15 / 0.01 (geographic frame) and 0.27 / 0.21 / 0.09
  (fixed-local-time frame); sign flips among |a|>0.3: 3.9% / 0.2% / 0%. Reading: v1 sits inside
  the GP's envelope on smoothness and amplitude, with ~1.4× the GP's gradients and more sign
  flips (structure that moves between 3-h frames). The plan's assumption that the anomaly
  co-rotates with the sun is wrong for the GP too: both change *less* in the geographic frame.
  Gate bounds (provisional, from the GP): fine-scale fraction < 0.01, grad_far ≤ 1.5× GP, sign flips
  < 10%. v1 passes. Needs many issue times to be a real bound; run it over a month later.

- 2026-09-05: **model_v1 (cross-attention-only decoder, 10 epochs) on 2025, paired**
  (`eval/model_v1.parquet`; still validated on 2025 → smoke number). Holdout foF2 primary 0.995 vs
  kernel 1.204 → +19.7% [+18.7, +20.9] (v0: +17.8%); flat across leads (0.98 → 1.03). Full mode vs
  the production GP: foF2 +4.5% [+3.4, +5.6], MUF +5.4% (v0: +1.8 / +2.7). RO vs IRI: foF2 +24.5%
  [+23.8, +25.4], hmF2 +13.1% (GP −3.4 / −6.8). σ calibration 66.2% / 93.3%. Maps smooth (coherence
  gate passed on the sample issue time). The decoder fix improved skill as well as maps.

- 2026-09-05: **model_v1 on production's 2023 holdouts** (3697 hourly issue times, one withheld station
  each, 79k paired rows; `eval/gp2023/model_v1.parquet`): foF2 RMSE 0.91 → 0.94 by lead vs GP
  0.98 → 1.13; skill vs GP +18.8% [+17.4, +20.0]; hmF2 +12.6%, MUF +18.0%; σ coverage 71.6 / 95.9%.
  **Caveat: 2023 is inside v1's training window**, so the withheld station's own rows at other
  issue times were training targets; this is not an out-of-sample year for the model the way it
  is for the GP. The clean version is the rolling protocol or a model trained through 2022 only.
  The 2025 comparisons (RO, full mode) are the out-of-sample ones.

- 2026-09-05: **model_v2 = v1 architecture trained on 2019-10 → 2024-09, validated on 2024-Q4 (no
  peek at 2025), 10 epochs.** On 2025, paired: indistinguishable from v1 — holdout foF2 primary
  0.998 vs 0.995 (v2 vs v1 −0.2% [−0.8, +0.4]); full mode vs GP +4.4% [+3.2, +5.5]; RO vs IRI
  +24.5%, hmF2 +13.2%. Calibration 63.8 / 91.8% (v1 66.2 / 93.3; both slightly narrow). So the
  2025 numbers are clean, out-of-sample results. **Phase 4a exit criterion met.** Reported
  numbers from here use v2 (or its successors trained with the same split).

- 2026-09-05: **rolling fine-tune, first pass** (`train/rolling.py`, chained, 6-month window incl. the
  training set, 300 steps at 2e-5 per month; the first attempt without `--history` overfit weeks of
  data and got worse every month). Jan–May 2025 so far, paired against static v2 on the same
  holdout rows: +0.8% [−0.4, +2.1] — neutral. Month-by-month it alternates (Jan −0.05 MHz, Feb +0.02,
  Mar −0.03, Apr +0.02). **Full year (Jan–Dec 2025): exactly neutral** — holdout foF2 +0.0% [−0.8, +0.9] vs static v2 on
  3.0M paired rows, RO −0.2% [−0.5, +0.1], vs GP +4.1% (static +4.4%); 7 months better / 5 worse,
  by ≤0.06 MHz either way; calibration unchanged. Monthly fine-tuning with a 6-month window neither
  drifts nor helps. **Decision:** the production update job re-scores monthly (safety and the
  rolling numbers) and retrains from scratch on all data quarterly; `--from-base` and a 12-month
  window remain cheap optional checks, not blockers. Rolling machinery validated.

- 2026-09-05: **Phase 4b, GloTEC pathway built** (`train/glotec_tokens.py`; `build_samples.py --glotec`
  emits `tok_g`; `model.py` adds a GloTEC projection + kind embeddings, `train.py --glotec` with 30%
  source dropout; predict/forecast pass the tokens). Tokens: every cell with qf>0 plus a quarter of
  qf=0 cells, at lags 0/−1/−6/−24 h, features per MODEL.md, ≤1200 per issue time. Old checkpoints
  load unchanged. **Protocol for the GloTEC gain (GloTEC exists only from 2025-05-12):** train v3
  on `train_to_2024q3` ∪ 2025-01 → 2025-09 (GloTEC tokens where the archive has the day), validate
  on 2025-10 → 2025-12, and train the identical model without `--glotec` on the same data as the
  control, so the paired difference on 2025-10 → 12 isolates the source rather than the extra
  training months. 2026-01 → 06 stays frozen for the final report. `samples/eval2025_g` (2025 with
  tok_g) building now.

- 2026-09-05: **GloTEC latency measured**: the NOAA daily file is updated ~16 min after a step's nominal
  time (file mtime 03:00:48 for the 02:45 step); with the 15-min run cadence the newest usable step
  is ~25 min old. Training tokens now assume 30 min (`LATENCY_MIN` in `train/glotec_tokens.py`),
  up from the 15 min copied from the ionosonde assumption; `eval2025_g` rebuilt. Andrew: live GloTEC
  is usually 25–40 min behind, so 30 is right; the ionosonde 15 min is close enough as is. The production
  fetch should log the observed latency so this becomes measured rather than assumed
  (Phase 5, alongside `inserted_at` for ionosondes).

- 2026-09-05: **Phase 4b GloTEC result: null.** v3_glotec vs v3_control (identical training, Oct–Dec 2025,
  paired): holdout +0.4% [−0.6, +1.5], full +0.7% [−0.5, +1.8], RO −0.1% [−0.6, +0.3]; validation
  1.053 vs 1.057. Stratified at RO by GloTEC coverage and lead: ≤0.6% even at qf5, 0–3 h. Cause is
  redundancy, not a plumbing fault (tokens confirmed present at prediction): at RO points in
  Oct–Dec the control model alone (1.25 MHz at qf5, 0–3 h) already beats GloTEC's own nowcast
  (1.38), and their errors correlate 0.71–0.73; averaging the two would gain only 3% at qf5 and lose
  at qf0. GloTEC's Phase 3 value was measured against IRI; against this model there is little
  left. Decision: keep the pathway (harmless under source dropout; useful if the ionosonde network
  thins further), no further GloTEC work now. Next lever is the model's own signal: hourly issue
  times (3× samples) and capacity; then the spot check with the same protocol.

- 2026-09-05: **outage capability of v3_glotec** (`predict.py --drop-iono` / `--drop-glotec`; Oct–Dec 2025,
  paired vs IRI): with GloTEC only, stations +13.7% [+12.2, +15.4] (1.27 MHz, flat with lead; ≈ the
  kernel's +15.3% with ionosondes), RO +12.2%, hmF2 +4.2%. With no observation tokens at all
  (indices + null token): stations +1.7%, RO +10.2% — the learned climatological correction to IRI
  at low latitudes alone is worth 10% at RO. Full model: +27.8% / +22.5%. So the fallback ladder is
  full > GloTEC-only ≈ kernel > nothing > IRI, and the model degrades gracefully. GloTEC-only vs
  no-inputs shows GloTEC contributes 14% at stations but only 2.5% at RO: the pathway learned to
  use GloTEC where training targets were dense (stations), less where they were sparse — the
  under-training side of the null 4b result. More GloTEC-era training data would help that.

- 2026-09-05: **v4 (hourly issue times, 45.7k samples, GloTEC pathway on) vs v3, Oct–Dec 2025,
  paired**: holdout +0.9% over v3_control [−0.2, +2.2] (v3_glotec +0.4%); val 1.047 vs 1.053/1.057;
  full vs GP +6.8% (v3: +6.4 / +5.5); RO vs IRI +23.0% (v3: 22.5 / 22.7). Three times the samples
  bought ≤1%: neighbouring issue times share observations, so they add little new signal. At this
  capacity the model is at the limit of what the ionosonde network + indices determine; more
  epochs/samples won't move it. Remaining levers: (a) one capacity run (d 256, 6 layers) to close
  that question; (b) sources whose errors are *independent* of the model's (GloTEC's were not);
  (c) ship it — v4 already beats production everywhere measured. Recommendation: (c) now, (a) and
  the spot check in the background.

- 2026-09-05: **skill by lead and distance** (`analysis/skill_by_lead_distance.py` + `skill_charts.py`,
  Oct–Dec 2025, artifact "Forecast Skill by Lead and Distance"). GP beats the model only at its own
  stations in the first ~4 h (0.74 vs 0.97 MHz at 1 h); the model is flat with lead everywhere else
  and its RO error grows 15% from <250 km to >4000 km from the nearest input station vs 50% for the
  GP. Likely cause of the short-lead gap: the 2048-token cap subsamples uniformly, discarding most of
  the last few minutes at busy stations. **Deferred (Andrew):** recency-weighted token sampling and a
  larger ionosonde token budget are tuning steps for after all sources are in.

- 2026-09-06: **Phase 5 service built** (`forecast/service/app.py`, `service/Dockerfile`, `deploy/prop-forecast.service`,
  `deploy/Task-Forecast.pm`, `deploy/scheduler.patch.md`; `FORECAST_PORT=5515`). `POST /forecast_24h`
  (run_id, model, glotec, holdout) reads inputs straight from Postgres, builds tokens with the training
  code, runs the checkpoint (GPU if present, CPU fallback ≈ 150 s for 25 hourly maps), and writes 25
  `assimilated` rows in assimilate's HDF5 layout (`/maps/{fof2,hmf2,mufd,md,foe,gyf}`, `/stdev/*` in
  real units, `/stationdata/{curr,pred}`, `/essn`, `/ts`; attrs record model/glotec/f107). foE from
  PyIRI's E layer, gyrofrequency from IGRF at 100 km as irimap does. No essn/pred/irimap dependency:
  the scheduler task `forecast_v2` has `parents => []`. Verified end to end against a throwaway
  Postgres seeded from the snapshot (2025-06-15 12:00, 30 stations, GloTEC on). Deployment: image
  `prop-forecast`, bind-mount `/home/prop/checkpoints` (`<model>/best.pt`, plus `iri_cache/` and
  `glotec/` subdirs), `--device=nvidia.com/gpu=all`; scheduler experiments `2026-09-v4` and
  `2026-09-v4-glotec` alongside the unchanged production run.

- 2026-09-06: **GP `/stdev` is the σ of log(value)** (Andrew); `eval/gp_maps.py` now emits linear σ ≈ value × σ_log.
  GP calibration, Oct–Dec 2025 (paired rows): foF2 coverage of ±1σ / ±2σ = 37% / 65% at assimilated
  stations and 44% / 72% at RO points (ideal 68 / 95); hmF2 59–61% / 86%. The GP is ~2× overconfident
  on foF2; the model's 64–70% / 92–95% is close to nominal. Earlier "GP σ not comparable" notes are
  superseded; the 2025 `gp_q*` files predate the fix (their σ column is still log-space).

- 2026-09-06: service cold-start diagnosed: PyIRI compute, not file I/O (0.4 s of I/O in a 20 s map
  build). Third PyIRI patch: the Legendre part of the SH basis depends on colatitude only, and PyIRI
  passes the same grid at every UT step, so it is now computed once per point and broadcast over
  time — cold 2° map build 20 s → 6.3 s at 4 threads, bit-exact. foE maps are cached per day like
  the IRI maps (`iri_cache.foe_points_cached`), and the service prewarms the next day's maps in a
  background thread after each run. Service logs a timing line per phase.

- 2026-09-06: renderer contract for `/stationdata/{curr,pred}` (from `renderer.py` + `plot.draw_dots`):
  `station.latitude`, `station.longitude`, `time` in **ms**, `cs` in **[0, 1]** (alpha = 0.2 + 0.6·cs;
  rows with cs < 0.249 or older than 1 h before the frame are dropped), metric columns. Service fixed
  (cs was raw 0–100 / missing on pred; time was seconds under pandas 3's µs resolution). pred `cs`
  is derived from σ: clip(1 − σ_fof2/3 MHz, 0.3, 1).

- 2026-09-06: **daily indices job** (`service/indices.py`, `deploy/prop-indices.{service,timer}`, 03:30 UTC):
  GFZ Kp/ap/F10.7 → `indices_daily`, SILSO → `ssn_monthly`, idempotent upserts (34.6k + 3.3k rows,
  verified twice against a throwaway Postgres). Until it runs, the forecast service falls back to
  the latest eSSN `sfi` as its F10.7 driver — the first live runs on 2026-09-05 used f107≈100 from
  that fallback where the trailing-81-day GFZ value was ≈129, i.e. a driver ~30 SFU below training
  conditions. Install the timer and run it once by hand. The service now also stamps
  `glotec_latency_min` into each map file (training assumed 30 min).

- 2026-09-06: live scoring path: `eval/gp_maps.py --experiment 2026-09-v4 --runs <fresh runs.parquet>` scores
  an experiment's archived runs into the replay schema as `live_<experiment>`, paired with production
  on the same targets. Monthly re-score = re-dump the DB (`snapshot.sh`), rebuild `ro_full` from the
  mirror, score production + experiments for the month, `report.py`. Experiment runs are archived to
  `/archive/<id>` after 3 days but deleted (not uploaded offsite) after 14; `deploy/scheduler.patch.md`
  has the one-line Cleanup.pm change to upload `2026-09-v4*` runs too. `gp_maps.py` reads either tree.

- 2026-09-06: **Status: live as experiments.** `2026-09-v4` and `2026-09-v4-glotec` run every 15 min
  beside production on the kc2g server (`prop-forecast` service, A4500), with the daily indices job
  installed. Model = v4 (ionosondes + indices, GloTEC pathway on in one experiment; no spots).
  Remaining, in order: (1) Cleanup.pm change so experiment runs are kept for scoring; (2) first
  live re-score after ~a month; (3) spot tokens (4b) with the GloTEC protocol; (4) capacity run;
  (5) coherence gate over a month of issue times; (6) deferred tuning: recency-weighted token
  sampling, and a **token-budget study** (Andrew, 2026-09-06): per-source cap (ionosonde 2048 /
  GloTEC 1200 / spot 3000) vs holdout skill vs train/inference runtime. Motivation: the GP still
  wins 0–4 h at its own stations, and the suspected cause is the ionosonde cap subsampling away
  the last few minutes at busy stations; inference is now ~7 s and VRAM-light, so there is room.
  Needs rebuilt samples per cap (MAX_TOK is baked in at build time) and the evaluate() chunking
  already in place; (7) promote to production when the live re-score
  confirms the offline result.

- 2026-09-06: **spot token pathway built** (`train/spot_tokens.py`; `build_samples.py --spots`; model kind 3 with a
  shared WSPR/FT8 projection + source one-hot; `train.py --spots`, 30% source dropout; predict
  `--drop-spots`). Token = (5° cell, hour, source): geometry, Δt, 10-band activity anomaly, 10-band
  SNR anomaly, log total, source one-hot (F=29). Anomalies vs a (cell, band, UT hour, month) median
  baseline from WSPR 2019–2023 (FT8: all its years, accepted mild leak); baseline support ≥6 hours.
  Hours ending in [T−23h, T−1h] (1 h latency); cap 3000 tokens, most recent hours first. Smoke-tested
  end to end. Aggregation of all WSPR years and FT8 2024/2026 running; chained: baseline → rebuild
  `train_1h_s` / `eval2025_gs` → `train_v5` / `val_v5` splits (same window as v3/v4, so v5-spots vs
  v5-control on Oct–Dec 2025 isolates the source).

- 2026-09-07: **v5 samples ready**: `train_1h_s` (43,497, 31 GB) and `eval2025_gs` (2,920) rebuilt with spot
  tokens; `train_v5` (45,681) / `val_v5` (736) splits as v3/v4. Every sample hits the 3000-token cap
  (≈ the last 9 h of cells); training years are WSPR-only, 2025 is half FT8, so the FT8 one-hot is
  learned from Jan–Sep 2025 only. Tarballs + listings on HF. Runs: `--glotec --spots` vs `--glotec`
  control, paired on Oct–Dec 2025.
- 2026-09-06: `AnomalyModel.forward` split into `encode()`/`decode()`. Validation in `train.py` now
  encodes once and decodes queries in 1024-chunks (val samples carry ~9k queries × 6.2k tokens; the
  unchunked B×heads×M×N attention OOMed the A4500). Same split applied to the map producers
  (`service/app.py`, `forecast.py`, `predict.py`), which had been re-running the encoder per
  8192-query chunk: live run 12.5→5.2 s with GloTEC, 10.0→4.7 s without (first post-rebuild run 7.5/7.0 s was cold-cache). Outputs
  bit-identical. Also: `kind_emb` load hook pads v3/v4 (2-row) checkpoints; `query_df` casts
  Decimal columns to float (regression from the read_sql fix).
- 2026-09-06: **v5_spots training** (train_v5/val_v5, `--glotec --spots`): val foF2 RMSE 1.073 / 1.054 /
  1.060 / 1.065 / … / 1.047 at epochs 1–6, train loss 0.25→0.02 by epoch 4 (memorisation; spot and
  GloTEC tokens get whole-source dropout only, no per-token keep). Epoch 6 pulls even with v4 (1.047)
  on the same window; epoch 2 and 6 checkpoints kept. Control run (`--glotec` only on train_v5) next,
  then paired predict/eval and `analysis/attention_mass.py` (learned relevance by token kind, added
  today with `decode(return_attn=True)`; v4 puts 0.86 of mass on ionosondes, ~4× GloTEC per token,
  flat across lead and distance). Candidate v5b fix: per-token 50–100% keep on tok_g/tok_s.
- 2026-09-06: **Phase 4b spots result: null, same shape as GloTEC.** v5_spots vs v5_control (identical
  train_v5/val_v5, `--glotec` ± `--spots`, Oct–Dec 2025, paired): holdout foF2 +0.4% [−1.1, +1.9],
  full +0.4% [−0.8, +1.6], RO −0.3% [−0.9, +0.3]; hmF2 at stations −1.2% [−2.3, −0.2]. vs GP full
  mode +6.6% (control +6.0%); RO vs IRI +22.9% (control +23.2%). By distance at RO: no bucket
  differs by >0.01 MHz, including >4000 km (1.340 vs 1.339). Outage views (v5_spots): drop spots
  −0.3%; spots-only (no iono, no GloTEC) +18.7% over IRI at holdouts vs +10.7% for the control with
  no inputs, and +15.0% vs +13.6% at RO. So spot activity carries real signal about the station
  anomaly field (about half of what the ionosondes give) but almost all of it is redundant with the
  ionosondes at the query points we can score. Attention mass (`analysis/attention_mass.py`, 120 val
  samples): iono 0.78, spot 0.15, GloTEC 0.07 of decoder cross-attention; per token spot 0.14 and
  GloTEC 0.16 of an ionosonde row; spot share rises from 0.13 near stations to 0.18 beyond 2000 km.
  Decision: keep the pathway (source dropout makes it free, and it is the best fallback if the
  ionosonde feed dies), no further spot-feature work. The two extra sources now behave as
  redundancy, not skill; the remaining levers are the ionosonde token budget / recency sampling and
  capacity.
- 2026-09-06: **single-receiver standalone test (W3USR, 41.40N 75.63W, May–Jun 2025).** Spot tokens built
  from that receiver's spots only (`wspr_aggregate.py --rx`, 790k WSPR + 1.58M FT8 on 1000–3000 km
  paths; ~450 tokens/sample vs ~3000 with the network), v5_spots with `--drop-iono --drop-glotec`,
  488 issue times, foF2 vs IRI, paired. Two baselines: the trained global one (single receiver
  reads as −0.42 activity anomaly everywhere = "bands dead") and the receiver's own May–Jun
  climatology. Global-wide: own-baseline +35.4% at holdout stations / +29.1% at RO vs no-inputs
  +34.5% / +30.4%, all-network-spots-only +36.7% / +33.6%, full model +50.2% / +38.8%. Global
  baseline is harmful at RO (+13.2%). Regional (own baseline, RMSE MHz, noinputs in brackets):
  RO <1500 km 1.149 (1.377), 1500–3000 km 1.201 (1.316), >3000 km no gain; holdout stations
  <1500 km 0.782 (0.895). Within 1500 km the one receiver matches the whole network's spot-only
  skill (1.172 / 0.776). Conclusion: a standalone receiver gives a real regional nowcast (~15% over
  the indices-only model within its single-hop footprint, nothing beyond), provided the activity
  baseline is the receiver's own. Files: `/kass/forecast/eval/w3usr/`, samples `w3usr_{gb,ob}`.
- 2026-09-06: **token-budget study machinery.** `build_samples.py --max-tok` (v6 sets at 8192 = effectively
  uncapped: 2025 has ~6.5k ionosonde rows/day, 2023 ~10k; 3-hourly 2019-10→2025-09 + val Oct–Dec,
  GloTEC + spots, building to `samples/train_v6_8192`, `val_v6_8192`; the first `train_v6`/`val_v6` build was accidentally capped at 2048 and is not the study set). `train.py --max-tok N [--recency-tau h]`
  subsamples at load time (uniform, or P ∝ exp(age/τ)), val deterministic per issue time;
  `predict.py` applies the checkpoint's budget automatically. Also v5b: per-token 50–100% keep on
  GloTEC and spot tokens in training (was whole-source drop only). Ladder to run (Andrew, GPU):
  `--max-tok 1024 / 2048 / 4096 / none` and `--max-tok 2048 --recency-tau 6`, scored on val_v6_8192 +
  Oct–Dec paired; watch epoch time and the 0–4 h full-mode gap to GP. Coherence gate over a month
  queued behind the builds: daily 00Z October 2025 maps, v5_spots vs GP grids, `/kass/forecast/maps/coh`.
- 2026-09-06: **coherence gate over a month** (daily 00Z, October 2025, 9 leads at 3 h, v5_spots with
  ionosonde inputs vs production GP grids; `/kass/forecast/maps/coh`, `eval/coherence_oct2025.parquet`).
  Means GP / v5: anomaly RMS 0.75 / 1.07 MHz; fine-scale fraction 0.000 / 0.001; grad_far 0.020 /
  0.061 MHz per 100 km (near 0.022 / 0.074); |anomaly| far 0.63 / 0.86; change per hour 0.12 / 0.26
  (geo), 0.16 / 0.39 (local time); sign flips 2.4% / 9.0% (max day 13.4%). Against the provisional
  bounds: fine-scale passes, sign flips pass on the mean but not on the worst days, grad_far is 3×
  the GP (bound was 1.5×). The GP is a smoothing kernel and the model's far-field skill is better
  (RO error grows 15% with distance vs the GP's 50%), so the gradient bound as calibrated penalises
  structure that scores; the sign-flip rate is the one to eyeball (3-h frame flicker). Andrew
  eyeballed the GIFs (`2025-10-DD_v5/fof2.gif`): more structure than the GP, not obviously wrong,
  plausibly real extrapolation. **Gate set** (monthly means over ≥30 daily issue times, vs the
  production GP on the same days): fine-scale fraction < 0.01, grad_far ≤ 3.5× GP, sign flips
  < 10%. v5 passes. Re-run with each promoted model.
- 2026-09-06: **long-distance spots via control points** (Andrew's idea): >3000 km paths contribute at
  the great-circle points 1500 km from each end (first/last F2 hop, as P.533 control points) instead
  of being discarded. One-sided signal (a spot proves both end regions supported the band; absence
  proves nothing), so it goes in separate token slots: `wspr_aggregate.py --control-points` →
  `{wspr,psk}_cp_hourly_*.parquet`, own baseline rows (source `wspr_cp`/`psk_cp`),
  `build_samples.py --spots --spots-cp` → 52-feature tokens (midpoint 21 + control-point 21 + 2
  presence flags + geometry/source), model `f_spot` from the checkpoint. Legacy 29-feature path
  verified bit-identical. Test plan: spots-only outage view (RO by distance from ionosondes,
  ocean/land), W3USR standalone with its long paths, one training run vs the v6 control.
- 2026-09-06: **coherence gate over a month** (October 2025, daily 00Z, 9 leads at 3 h, v5_spots vs
  production GP, `/kass/forecast/maps/coh`, `eval/coherence_oct2025.parquet`). Means over 31 days,
  v5 / GP: anomaly RMS 1.07 / 0.75 MHz; fine-scale (k>20) fraction 0.0007 / 0.0000; grad_far
  0.061 / 0.020 MHz per 100 km (per-day ratio median 3.3×, range 1.6–4.3×); grad_near 0.074 /
  0.022; |anomaly| far 0.86 / 0.63; RMS change per hour 0.26 / 0.12 (geo), 0.39 / 0.16 (local
  time); sign flips 9.0% / 2.4% (worst day 13.4% / 9.6%). Against the provisional bounds:
  fine-scale passes by two orders; sign flips pass on the mean, fail on the worst 3 days; grad_far
  fails (3.3× vs the 1.5× bound). The gradient is at large scales (fine-scale ~0) and scales with
  amplitude: per MHz of anomaly the model has 0.057 vs the GP's 0.027, i.e. 2× on every day, the
  same ratio as the single June day (where the GP happened to be at 1.4×). The GP is uniformly
  smooth (near ≈ far gradient); the model has structure near stations decaying outward. Since
  the model beats the GP by 23% at RO points in the same window, the extra large-scale structure
  is real signal, not noise. Decision: the 1.5×-GP gradient bound was mis-set on one day; replace
  with amplitude-normalised gradient ≤ 2.5× GP and sign flips < 15% worst-day, keep fine-scale
  < 0.01. v5 passes the revised gate. Eyeball worst day 2025-10-24.
- 2026-09-06: **attention on SDPA + hourly token pooling** (ahead of the token-budget ladder; both change what
  the budget means). `train/model.py`: `MultiheadSDPA` (q/k/v/out Linears, padding mask as a broadcast
  additive bias so the memory-efficient kernel applies; attention memory O(N) instead of the fp32
  (B,heads,Nq,Nk) scores nn.MultiheadAttention materialises with a key_padding_mask) in the encoder
  (`EncLayer`, pre-LN, ReLU FFN as before) and the cross-attention decoder; explicit path kept for
  `need_weights`. Old checkpoints load through `_compat` (in_proj split, encoder.layers.i rename,
  tok_in zero-padded, kind_emb padded): v5_spots reproduces the pre-change outputs to 2e-6.
  `build_samples.py --pool` (`pool_hourly`): one token per (station, hour of age) with mean time /
  values / cs and a 17th feature log1p(n)/3 (F_TOK 16→17; unpooled rows carry n=1). 2025-10-15:
  5454 rows → 633 tokens (28 stations, n̄ 8.6); 2023-03-01: 7116 → 1002. Checkpoint args carry
  `pool`; service/forecast.py/predict.py apply the matching tokenisation; 16-wide samples are padded
  at load. CPU encoder time B=2: 0.06 s at 2k tokens, 1.1 s at 10k. Pooled sets `train_v6p`/`val_v6p`
  queued behind the unpooled `*_v6_8192` rebuild. Ladder now: pooled uncapped (~1k tokens, cheapest
  and keeps every station-hour) vs unpooled 8192 vs unpooled 2048 (= v5 conditions).
- 2026-09-07: **v6_pooled** (train_v6p/val_v6p: 3-hourly, hourly-pooled ionosonde tokens ≈1k, uncapped, GloTEC +
  spots, SDPA attention, per-token source dropout; A40 560 s/epoch = 32 ms/sample vs 50 for v5; val
  1.042). Oct–Dec 2025 paired vs v5_spots: holdout foF2 +0.5% [−0.7, +1.6], full vs GP +7.7%
  (v5 +6.6%), RO vs IRI +23.9% (v5 +22.9%), hmF2 at holdouts −3.5% RMSE (25.3 vs 26.3 km), σ coverage
  64/92% (v5 61/90%). Every lead and distance bucket equal or better. The 0–1 h full-mode gap to the
  GP at its own stations barely moves (0.965 vs 0.979 MHz; GP 0.737), so the token cap was *not* its
  cause; that gap needs something else (a station-identity / last-value pathway, or the GP's own
  persistence at 0–2 h) and is deferred. Decision: pooling adopted as the default tokenisation
  (same skill, all station-hours kept, 1.6× cheaper, ~9× fewer ionosonde tokens). Unpooled 8192
  and 2048 runs still to come for the ladder record.
- 2026-09-07: **pooling is the default; v7 rebuilt pooled.** Andrew: pooled 2048 beat unpooled 4096, so
  `--pool` is the standard ionosonde tokenisation. The unpooled `train/val_v6_8192` and
  `train/val_v7_8192` sets are abandoned (still on disk, delete when convenient). Building
  `train_v7p`/`val_v7p` = the v6p recipe (pooled, cap 8192, 3-hourly, GloTEC + spots) + `--spots-cp`,
  first pass CRC-clean (Andrew: the v6p corruption was a one-off, no scans in future builds); `w3usr_cpp` for the
  standalone test. Comparison: v7p vs v6_pooled at the same `--max-tok`; note the 3000 spot cap is
  shared between midpoint and control-point tokens (equal-budget test; if v7p loses, rebuild with
  a 6000 spot cap before concluding).
- 2026-09-07: **v7p (control-point spots, pooled) vs v6_pooled, Oct–Dec 2025, paired, foF2.** Val 1.032 vs
  1.042. Holdout +1.0% [−0.0, +2.1]; full +0.1% [−1.0, +1.2]; RO −1.3% [−1.8, −0.9]; vs GP full
  mode +7.5% (v6p +7.7%). Outage views (no iono, no GloTEC): spots-only +20.5% over IRI at holdouts
  vs v6p +18.6%, and at RO +15.8% vs +1.6% (no-input references +11.8 / +13.4 at holdouts, +8.8 /
  −3.4 at RO, so v7p's spot-attributable gain is +8.7 / +7.0 pts vs v6p's +5.2 / +5.0). Spots-only
  RO by distance to nearest ionosonde: v7p 1.28→1.48 MHz (<250 km → >4000 km) vs v6p 1.41→1.76, i.e.
  control points carry real ocean information. Attention mass: spot 0.42, iono 0.37 (pooled, ~1k
  tokens), GloTEC 0.21. **W3USR standalone with control points** (`w3usr_cpp`, v7p, own baseline):
  stations +38.1% over IRI vs its no-input +34.8% (midpoint-only v5: +35.4 vs +34.5); RO +30.1 vs
  +30.9. By distance from W3USR at RO (cp / no-input / midpoint-only v5): <1500 km 1.247 / 1.395 /
  1.149; 1500–3000 1.230 / 1.334 / 1.201; 3000–6000 1.228 / 1.279 / 1.258; >6000 1.337 / 1.306 / 1.357.
  Reading: control points spread the receiver's information out to 3000–6000 km and lift the global
  station number, but the near field is worse than midpoint-only, and the RO loss with full inputs
  points the same way: the shared 3000 spot cap lets control-point tokens displace midpoint tokens
  (78% of w3usr_cpp spot tokens are control-point-only, 65% in train_v7p). **Next:** rebuild v7p
  with the spot cap at 6000 (or 3000 per slot) and retrain once; judge control points on that.
- 2026-09-07: **v7p skill dashboard** published ("v7p Forecast Skill",
  https://claude.ai/code/artifact/2a42ab5f-783d-4fca-9aec-4cafe9f3fdc1; generator
  `analysis/skill_dashboard.py` from `skill_model_v7p.json` + a sources json). Headline, Oct–Dec 2025
  paired: held-out stations +7.8% [+3.6, +11.5] over the GP, all stations +7.5%, RO +22.9% over IRI,
  σ coverage 63/91%. Source-availability (skill over IRI, holdout / RO): everything +29.0 / +22.9;
  no spots +29.0 / +22.7; no GloTEC +29.0 / +23.0; ionosondes only +29.0 / +22.9; spots only
  +20.5 / +15.8; GloTEC only +18.1 / +12.0; indices only +11.8 / +8.8. With ionosondes present the
  other two sources are worth nothing measurable; each alone recovers half to two thirds of the
  ionosonde skill.
- 2026-09-07: **service compatibility.** `/stationdata/{curr,pred}` now carry `station.id` (assimilate's
  json_normalize schema; prop-cosmic reads it). Verified v4 (16-feature, unpooled) and v7p (pooled,
  52-wide spots) both load and tokenise per their own checkpoint args. v7p in the service runs
  without spot tokens (= the "no spots" row, identical skill with ionosondes present). **Next
  (Andrew): live spot fetching for the service** (hourly WSPR + pskreporter aggregates reachable
  from the pod, then `spot_tokens` at forecast time) so the spots-only outage path exists in
  production; a v7p experiment goes live without it first.
- 2026-09-07: **Muon optimizer option** (`train/muon.py`, vendored from arodland/SSTVAE `sstvae/muon.py`;
  `train.py --optimizer muon`). Weight matrices go to Muon (Newton-Schulz orthogonalised nesterov
  momentum, `match_rms_adamw` lr scaling so `--lr` and the cosine schedule keep their AdamW
  meaning); biases, LayerNorm gains, the null token, kind embeddings and the output head stay on
  AdamW. 1.29M of 1.30M params orthogonalised. Smoke-trained on CPU. Test (Andrew, GPU): v7p
  recipe with `--optimizer muon` vs the AdamW v7p, same seed/flags: convergence speed (epoch of
  best val) and best val / paired Oct–Dec skill.
- 2026-09-07: **v7p_muon (v7p recipe, `--optimizer muon`), Oct–Dec 2025, paired, foF2.** Val 1.014 at epoch 4
  (v7p AdamW 1.032; train loss then keeps falling while val drifts up, so epochs 5–10 overfit).
  vs v7p: holdout +1.8% [+0.6, +3.0], full +3.3% [+2.4, +4.5], RO +0.5% [+0.0, +0.9]; vs GP full
  mode +11.2% [+8.7, +13.6] (v7p +7.5), holdout +7.1% (v7p +7.8, same CI); 0–1 h full-mode 0.879
  vs 0.968 MHz (GP 0.737). σ 63/92%. RO by distance better in every bin but the farthest.
  **But the degraded modes regressed:** no inputs +1.6% over IRI at holdouts / −17.9% at RO (v7p
  +11.8 / +8.8); spots-only +14.7 / +5.4 (v7p +20.5 / +15.8); GloTEC-only +16.3 / +10.8 (v7p +18.1 / +12.0);
  ionosondes-only +29.7 / +22.9 and the no-spots / no-GloTEC rows sit on "everything", as before. At epoch 4 the source-dropout
  fallbacks (indices-only, spots-only) are not yet learned; early stopping on the all-sources
  metric picks a checkpoint that is best-case better and outage worse. Next: (a) 8-epoch cosine
  so the anneal lands at the peak; (b) if the fallbacks are still weak, raise whole-source drop
  probabilities (P_DROP_GLO/SPOT 0.3, and add an explicit all-sources-dropped fraction) so they get
  enough steps; (c) Muon on v7p6 for the control-point verdict. Muon is the default optimizer
  candidate once (b) is settled. Andrew: no-inputs skill is not a goal (only forecasts with at
  least one observation type count); spots-only and GloTEC-only are the outage views that matter.
  8-epoch Muon run: best val 1.033 at epoch 6, worse than the 20-epoch schedule's epoch-4 1.014
  (the short anneal did not help; the epoch-4 peak was a high-LR effect, not an under-annealed
  one). A 20-epoch run at lr 1e-4 did not pay off either. **Decision (Andrew, 2026-09-08): stay on
  AdamW for now**; Muon stays available (`--optimizer muon`) for a later look, with the epoch-4
  best-case result (1.014, +11% over GP full mode) as the thing to recover without losing the
  single-source fallbacks.
- 2026-09-08: dashboard gained a "no ionosondes" (spots + GloTEC) row: +19.9% / +14.4% over IRI at
  holdouts / RO, *below* spots-only (+20.5 / +15.8) and only a little above GloTEC-only (+18.1 /
  +12.0). **TODO (Andrew): revisit GloTEC filtering.** Adding GloTEC to spots costs skill, so some
  GloTEC tokens are worse than nothing; candidates are the qf=0 cells (¼ of them are sampled in),
  the −24 h lag, and the latency jitter. Test by rebuilding tokens with qf≥1 only / fewer lags and
  re-scoring the spots+GloTEC and GloTEC-only outage views. Not urgent: with ionosondes present
  GloTEC is neutral.
- 2026-09-08: **v7p6 (control points, spot cap 6000) vs v7p, Oct–Dec 2025, paired, foF2: control
  points adopted.** All sources: a wash (holdout −0.5% [−1.5, +0.4], full −0.4%, RO −0.3%; vs GP
  +6.9% both views; val 1.037). Every degraded mode improves (skill over IRI, holdout / RO):
  spots-only +23.4 / +16.8 (v7p +20.5 / +15.8); spots+GloTEC +23.7 / +17.5 (v7p +19.9 / +14.4);
  GloTEC-only +22.8 / +18.5 (v7p +18.1 / +12.0). Spots-only RO by distance to the nearest
  ionosonde 1.25→1.47 MHz vs v7p 1.28→1.48. The spots+GloTEC < spots-only inversion is gone (the
  GloTEC filtering TODO stays, lower priority). **W3USR standalone** (`w3usr_cpp6`): stations
  +41.3% over IRI (cp@3000 +38.1, midpoint-only +35.4, no inputs +34.7); RO by distance from the
  receiver 1.125 / 1.189 / 1.269 / 1.385 MHz (<1500 / 1500–3000 / 3000–6000 / >6000) vs no inputs
  1.197 / 1.242 / 1.248 / 1.342: the near field is back and better than midpoint-only (1.149), and
  the receiver now helps out to 3000 km; beyond 3000 km a lone receiver is slightly worse than
  nothing (standalone caveat, not a network one). Decision: v7p6 is the current model; dashboard
  republished from it. The GloTEC-only jump (+18.1→+22.8 at holdouts) with unchanged GloTEC
  tokens says run-to-run variance in the fallbacks is a few points; treat single-source
  differences under ~3 points as noise.
- 2026-09-08: **training stability (Andrew).** Much of the checkpoint-to-checkpoint spread in the
  single-source rows (and the epoch-4 Muon peak) is luck of where an epoch boundary lands, not real
  variation. Wanted: a training procedure that reliably does well on *all* the skills. Cheap
  levers, in order: (1) validate every N steps rather than per epoch and keep an EMA of the weights
  (EMA checkpoints average out the epoch-boundary lottery and usually help σ calibration too);
  (2) select the checkpoint on a composite val metric (all-sources foF2 plus the spots-only and
  GloTEC-only foF2 on a fixed val subset) instead of all-sources alone, so a fallback regression
  cannot slip through; (3) seeds: two seeds of the same recipe give the noise floor for free.
  Not started; v7p6 goes to the service first, then the live spot loader.
- 2026-09-08: **live spot loader, design.** Constraints (Andrew): /kass is training data only, never
  live serving; the fetch runs on the server inside the pod; local fast storage under /home/prop
  (or postgres / a sidecar ClickHouse) is available; a public ClickHouse HTTP endpoint serves
  `wspr.rx` and `pskreporter.rx` as parquet with bulk-fetch permission (`select * ... where time >
  $watermark format Parquet`). Design (smallest thing that reproduces training tokens exactly):
  one script `service/spots.py` on an hourly systemd timer (`prop-spots.timer`, :03), same
  pattern as the indices job. (1) Incremental pull per source from the HTTP endpoint with a
  watermark file, 10-min overlap, dedupe (wspr on `id`, pskreporter on time/tx/rx/band/mode),
  HF bands only, into daily raw parquet under `/home/prop/spots/raw/{wspr,psk}/YYYY-MM-DD.parquet`;
  3-day retention locally. (2) Re-aggregate the trailing 26 h with the existing
  `wspr_aggregate.aggregate()` in both modes (midpoint, control points) for both sources into
  `/home/prop/spots/agg/{wspr,psk,wspr_cp,psk_cp}_hourly_live.parquet`, complete hours only.
  (3) The service gets `SPOT_AGG_DIR=/home/prop/spots/agg` and `SPOT_BASELINE` pointing at
  `spot_baseline_cp.parquet` shipped once next to the checkpoints; `spot_tokens(t0)` then runs
  unchanged at forecast time, tokens identical to training modulo arrival timing. No ClickHouse
  sidecar, no postgres tables: duckdb over local parquet is the training path. Optional: copy the
  daily raw files to /kass write-only for future training. Freshness is not tight: tokens use
  hours ending ≤ T−1 h, so an hourly pull at :03 is enough for the :10/:25/:40/:55 runs.
- 2026-09-08: **live spot loader built** (`service/spots.py`, `deploy/prop-spots.{service,timer}`, notes in
  `deploy/scheduler.patch.md`). Every 15 min at :08/:23/:38/:53 (Andrew: same cadence as the ionosonde
  fetch; small frequent pulls are friendlier to the database): incremental `SELECT ... WHERE time >
  watermark` from https://wd1.wsprdaemon.org (no auth, ~5M WSPR + ~30M FT8 rows/day), 10-min
  overlap, dedupe at aggregation (WSPR on id, FT8 on time/band/mode/rx/tx), HF bands + FT8 only,
  raw per-pull parquet under `/home/prop/checkpoints/spots/raw` (3 days), trailing-26 h complete-hour
  aggregates in all four modes via the training `aggregate()` into `agg/*_hourly_live.parquet`.
  Measured locally: cold backfill 27 s, incremental 3 s WSPR / 15 s FT8. Service: `spot_tokens`
  at forecast time when the checkpoint has spots and `agg/` exists, cap `FORECAST_SPOT_CAP`
  (6000), CP slots from `f_spot`, gated on a `spots` form param like `glotec` so experiments can run v7p6 with and without spots side by side; attrs `spots`, `n_spot_tokens`, `spot_latency_min`. End-to-end check on
  live data: 6000×52 tokens, 65% control-point-only, activity anomaly mean 0.00 against the
  shipped baseline, Δt −1.5…−4.5 h (recent-first truncation at the cap keeps ~3 h of history,
  same as training). Andrew deploys (baseline copy, image rebuild, timer, restart).
  **Later (Andrew):** spot-token selection under the cap. Most of the value is in recent hours, but
  reserve ~10% of the budget for older hours (the 24 h recurrence) instead of pure recent-first
  truncation; a stratified pick (90% newest-first, 10% uniform over the rest of the 24 h) in
  `spot_tokens`, rebuilt samples, one training run. Part of the token-budget study.
- 2026-09-08: **live spot tokens running in production.** First run produced 0 tokens (baseline file not yet
  copied; the service now logs baseline/aggregate presence when the token set is empty). Fixed by
  shipping `spot_baseline_cp.parquet`. Experiments `2026-09-v7p6` and `2026-09-v7p6-spots` run side
  by side with GloTEC on; first live re-score after ~a month with `gp_maps.py --experiment` on all
  four experiments (v4, v4-glotec, v7p6, v7p6-spots) against production.
- 2026-09-08: **fresher spot bins (Andrew): hour bins ending at T−15 min instead of clock hours.** Live
  data reaches T−10 min from wd1 (T−5 min standalone), so stopping at the last full clock hour wastes
  up to an hour. Implemented behind `SPOT_RES=5` / `build_samples --spot-res 5`: `wspr_aggregate.py
  --res 5` writes 5-min bins (`{src}[_cp]_5min_<year>.parquet`, plus `snr_sum` so bins roll up);
  `spot_tokens` at res 5 forms 24 bins [E0−(k+1) h, E0−k h) with E0 = floor5(T−15 min), SNR = mean
  over spots, baseline keyed by the UT hour of the bin midpoint (`spot_baseline_cp5.parquet`, hourly
  roll-ups of the 5-min bins, so activity/SNR semantics match); the loader (`service/spots.py`)
  and service follow `SPOT_RES`. Hourly path verified bit-identical; res-5 live tokens land at
  Δt = −15, −75, −135 … min. Chain running: 5-min aggregates for all years (both modes), res-5
  baseline, `train_v8p6`/`val_v8p6` (= v7p6 recipe + `--spot-res 5`). Then one training run vs
  v7p6. Deploy note: a res-5 checkpoint needs `SPOT_RES=5` in the loader and service env.
- 2026-09-08: **EMA weights in training** (`train.py --ema 0.999`, first of the stability levers). Per-step
  EMA of all parameters with decay ramping from 0 (`min(ema, (1+step)/(10+step))`); at 0.999 the
  horizon is ~1000 steps ≈ half an epoch at bs 8. Each epoch validates raw and EMA; `best.pt` is
  the best EMA checkpoint (what the service and predict.py load, format unchanged), `best_raw.pt`
  the best raw one, both with `epoch` recorded. Also: samples now carry `spot_res`, checkpoints
  record it, and the service refuses a checkpoint whose spot resolution differs from `SPOT_RES`.
  Test (Andrew): v7p6 recipe + `--ema 0.999` vs v7p6; look at the epoch-to-epoch val spread of the
  EMA column vs the raw one, and the single-source rows.
- 2026-09-08: 5-min aggregation chain notes. (1) FT8 2025 control points OOM'd at duckdb's 48 GB
  (`wspr_aggregate.py --memory` added; rerun at 96 GB). (2) A transient NFS read error truncated
  several monthly part files; while cleaning up I ran a validity check that misreported every part
  file as bad and deleted the `*.parts/` intermediates for all years. The merged per-year 5-min
  files were verified intact afterwards (row counts and max hour per file), so the only cost is that
  a rerun of an already-finished year would recompute rather than skip. (3) **pskreporter mirror
  gaps are real on wd1 too**: the mirror's empty daily files (2024-12 ×25, 2025-01 ×6, 2025-02 ×10,
  2025-06 ×13, 2025-10 ×7, 2025-11 ×12, 2025-12 ×30 (only Dec 1 present), 2026-01 ×25, 2026-02 ×10)
  match wd1's monthly counts (Dec 2025: 29M rows vs 560–790M for Oct/Nov), so December 2025 FT8 is
  essentially absent from the Oct–Dec validation window in every spots model so far; WSPR covers
  it. Worth remembering when reading the spots-only rows.
- 2026-09-08: **GloTEC fetch failure handling.** After 00:00 UTC NOAA's new daily file is not yet
  published (404); `glotec_tokens` then had no lag-0 step and crashed (`keep0` unbound), and `_load`
  cached the missing day as absent for the rest of the process. Now: a step is chosen as the newest
  at or before the nominal time, looking in that day's file then the previous day's, up to 3 h stale
  (`STALE_MAX_H`); the Δt feature carries the step's true age (identical cells/values to the old
  nearest-step rule when the nominal step exists, Δt within the ±20 min training jitter); if lag 0
  has nothing usable the model runs without GloTEC (empty tokens, logged) instead of failing the
  run; missing days are no longer cached. Verified on the mirror: T=00:30 after the last file →
  1200 tokens with lag-0 age 1.25 h; T=04:00 → no tokens, run proceeds.
- 2026-09-08: **v7p6m (v7p6 recipe + Muon + EMA 0.999, best EMA at epoch 6, val 1.030) vs v7p6, Oct–Dec
  2025, paired, foF2.** Best case better: holdout +0.7% [−0.4, +1.9], full +1.9% [+0.8, +3.1], RO
  +2.4% [+1.8, +3.2]; vs GP full mode +9.0% (v7p6 +6.9), holdout +7.0% (same); hmF2/MUF +0.5–1.2%;
  σ coverage improves to 65/93% (v7p6 63/92); RO by distance better in every bin (1.149→1.318 vs
  1.193→1.343). **Fallbacks worse again, same shape as v7p_muon:** skill over IRI holdout / RO,
  spots-only +18.4 / +11.6 (v7p6 +23.4 / +16.8), GloTEC-only +11.9 / +13.7 (+22.8 / +18.5),
  spots+GloTEC +17.4 / +12.8 (+23.7 / +17.5), no inputs +0.9 / −20.9 (+12.2 / +0.8). EMA did not
  change this: Muon reaches the all-sources optimum by epoch 5–6 while the source-dropout
  fallbacks are still immature, and selection on the all-sources metric locks that in. Decision:
  v7p6 stays the production candidate. To make Muon usable: checkpoint selection on a composite
  val metric (all-sources + spots-only + GloTEC-only foF2), the second stability lever, and/or
  more whole-source dropout so the fallbacks get steps; then re-run.
- 2026-09-08: **the fallback knob: ionosondes were never dropped whole in training.** `Samples` kept
  50–100% of ionosonde tokens (min 5) and dropped GloTEC/spots whole at 30%, so spots-only,
  GloTEC-only and no-input forecasts were never a training case; they work only by generalisation,
  which is why they converge late, vary by several points between runs, and lose out when a faster
  optimizer reaches the all-sources optimum early. Andrew: rather than a composite selection
  metric (the primary may already be past its best by the time the fallbacks catch up), a knob
  that makes all modes converge together. Added `train.py --p-drop-iono` (whole-source ionosonde
  drop, default 0 = old recipe) plus `--p-drop-glotec` / `--p-drop-spots` (default 0.3). At
  0.2/0.3/0.3 independent: spots-only 4.2%, GloTEC-only 4.2%, no inputs 1.8%, no ionosondes 9.8%
  of samples train the fallbacks directly every epoch. Verified: empty-ionosonde and all-empty
  batches collate and train (null + global tokens always present). Test (Andrew): v7p6 recipe +
  Muon + EMA + `--p-drop-iono 0.2`; if the fallbacks now track the primary, the second question
  (overfitting after epoch ~5) gets its own knob: weight decay / dropout 0.1→0.2. Order (Andrew):
  settle the fallback-convergence question on the v7p6 samples first (`v7p6m_di` running
  2026-09-08), *then* train v8p6 (fresher spot bins, samples building) with the settled recipe.
- 2026-09-08: **v8p6 samples ready** (`train_v8p6` 17,419 / `val_v8p6` 736; v7p6 recipe + 5-min spot bins,
  `spot_res=5` recorded in every sample; tokens at Δt −15/−75/−135… min, activity anomaly mean 0.00
  against `spot_baseline_cp5.parquet`). Queued behind the fallback-convergence run per Andrew.
  Also fixed today: `--p-drop-iono` samples skipped the 1024-query cap (mis-nested block), which
  made every batch with a dropped sample run the decoder at ~9× queries (+50% epoch time) and
  over-weighted those samples; the first `v7p6m_di` epoch ran with that bug.
- 2026-09-09: **v7p6m_di (v7p6 + Muon + EMA + `--p-drop-iono 0.2`), Oct–Dec 2025, paired, foF2: the knob
  works, and it reveals something.** Best case vs v7p6: holdout +0.3% [−0.7, +1.3], full +0.6%
  [−0.3, +1.6], RO +1.1% [+0.5, +1.7] (v7p6m had +0.7 / +1.9 / +2.4, so ~half of Muon's edge is
  spent); vs GP full +7.7%, holdout +7.1%; σ 66/93%; val 1.034 at epoch 5. Fallbacks, skill over
  IRI holdout / RO: spots-only +25.6 / +22.4 (v7p6 +23.4 / +16.8), GloTEC-only +23.5 / +20.5
  (+22.8 / +18.5), spots+GloTEC +25.4 / +22.3 (+23.7 / +17.5), **no inputs +24.9 / +21.9 (v7p6 +12.2 /
  +0.8)**. RO by distance, no-inputs: 1.198→1.358 MHz vs IRI 1.437→1.793.
  **Reading:** once the no-observation mode is trained, indices + geometry + local time alone
  correct IRI by ~22% at RO and ~25% at held-out stations; observations add ~1–2 points at RO and
  ~5 at stations on top. So most of the model's far-field / RO skill over IRI is a learned
  climatological correction of IRI (bias by region, local time, season, F10.7), not extrapolated
  weather; that was always inside the full model, just never measurable because the no-input
  mode was untrained. Consequences: (1) the fair reference for "what is a source worth" is now the
  model's own no-inputs mode, not IRI, and the dashboard's source table should show skill over
  that too; (2) spots/GloTEC add ~1–3 points over learned climatology at RO, ~1 at stations —
  consistent with their null effect beside ionosondes; (3) `--p-drop-iono 0.2` is the new default
  recipe (fallbacks converge with the primary; single-source rows now all within ~2 points).
  Remaining Muon question is the best-case give-back (half); try `--p-drop-iono 0.1`, and the
  overfitting knob (dropout 0.2 / weight decay) so more epochs are usable. Then v8p6.
- 2026-09-09: overfitting knobs exposed: `train.py --dropout` (attention/FFN dropout in every layer,
  default 0.1 = v1..v7) and `--wd` (decoupled weight decay for AdamW and Muon, default 0.01).
  Both recorded in the checkpoint args. Runs (Andrew): v7p6m_di recipe with `--dropout 0.2`, and
  with `--p-drop-iono 0.1`; judge on val curve flatness after epoch 5 plus the paired report.
- 2026-09-09: **v7p6m_di2 (= v7p6m_di + `--dropout 0.2`), Oct–Dec 2025, paired, foF2.** Best EMA at
  epoch 8 (di: 5; dropout bought three more usable epochs), val 1.032. vs v7p6: holdout +0.5%
  [−0.6, +1.5], full +1.1% [+0.2, +2.1], RO +1.5% [+0.8, +2.3] (di: +0.3 / +0.6 / +1.1; v7p6m
  +0.7 / +1.9 / +2.4); vs GP full +8.3%, holdout +7.0%; σ 63/92% (di 66/93); hmF2 ±0. RO by
  distance 1.150→1.335 (between v7p6m and di). Fallbacks stay converged (holdout / RO over IRI):
  spots-only +24.1 / +22.1, GloTEC-only +22.1 / +19.4, spots+GloTEC +24.3 / +22.3, no inputs
  +24.6 / +21.6 — all within ~2 points of di. **Decision: recipe = Muon + EMA 0.999 +
  p-drop-iono 0.2 + dropout 0.2; v7p6m_di2 is the current best all-round model** (best case
  above v7p6 on every view, fallbacks intact). Next: train v8p6 (fresher spot bins) with this
  recipe and pair against v7p6m_di2; the wd knob stays untried.
- 2026-09-09: **dashboard republished from v7p6m_di2** (same URL) with a second reference in the source table:
  skill over the model's own indices-only mode. Over that learned climatology (holdout / RO):
  everything +5.8 / +2.9, ionosondes only +5.1 / +2.9, no spots +4.5 / +2.5, spots-only −0.5 / +0.6,
  spots+GloTEC −0.3 / +1.0, **GloTEC-only −3.6 / −2.7**. So beside ionosondes, spots are worth ~1
  point at stations and GloTEC nothing; alone, spots are neutral against learned climatology and
  GloTEC is *harmful*. That raises the GloTEC filtering TODO from low to medium priority: the
  GloTEC pathway as built subtracts skill whenever it is the only source.
- 2026-09-09: **negative marginal skill (GloTEC-only below indices-only) — checkpoint noise or a limit?**
  Andrew: the single-source rows have gone back and forth, so treat it as possibly checkpoint
  noise. Training-side levers, in order of cost: (1) *see it during training*: `train.py
  --val-modes` prints per-epoch val foF2 for ionosondes-only / GloTEC-only / spots-only / no
  inputs from the checkpoint that best.pt would hold (4 extra val passes/epoch, ~5 min on GPU) —
  built; if the GloTEC-only column swings by several points between epochs while the others are
  steady, it is noise and checkpoint selection can weigh it; (2) *consistency regularisation*:
  for a fraction of batches, run a second forward with a source dropped and penalise divergence
  of its (μ, logσ) from the full-input prediction (detached) — directly discourages a lone source
  from pushing the answer away from what the full picture says, which is what negative marginal
  skill is; costs ~+25% per epoch at 1-in-4 batches; not built; (3) more single-source samples
  (`--p-drop-iono` 0.2→0.3) so those modes get more steps; (4) the GloTEC filtering TODO
  (qf=0 cells, −24 h lag) if the column is consistently low rather than noisy. Recommendation:
  add `--val-modes` to the v8p6 run and decide between (2) and (4) from its log.
- 2026-09-09: **GloTEC qf=0 cells excluded** (Andrew: qf=0 means GloTEC had no observations in the cell and
  is extrapolating from IRI; extrapolating from observations is our job, not someone else's).
  `glotec_tokens` keeps qf>0 only (was: all qf>0 plus a 25% sample of qf=0); `train.py`,
  `predict.py` and `attention_mass.py` drop qf=0 rows (one-hot column 7) from samples built
  before today, so v8p6 trains on the filtered set without a rebuild; the service follows the
  code. Older checkpoints saw qf=0 tokens at 25% and now see none, an in-distribution subset.
- 2026-09-09: **sanity check on "indices-only explains most of the skill" (Andrew: too much from F10.7 alone,
  too little from observations, no ~4 h decay).** Oct–Dec 2025, full-mode stations unless noted.
  (1) *Our PyIRI reference is biased*: pred−truth +0.71 MHz (by UT −0.37…−0.93; Nov −0.90), RMSE
  1.445, while production `irimap_prod` (Fortran IRI, eSSN driver) is −0.10 / 1.321. Removing a
  0.7 bias alone takes 1.445→1.26, i.e. ~13 of the 25 "climatology" points are our baseline's bias.
  Against irimap_prod the indices-only mode is still +18.9% (1.094 vs 1.321) and the full model
  +25.0%: a learned per-location climatology with daily F10.7/ap beats the global URSI maps, which
  is plausible for stations (station-specific means) and less certain at RO (kind-flag check
  running: RO scored as map pixels). (2) *Observation skill relative to learned climatology has the
  expected shape but not the expected size*: held-out stations 1 h +11.2% → 6 h +8.9 → 12 h +4.5 →
  24 h +2.1 (half-life ~8 h); own stations +14.8 → +9.2 → +5.8 → +3.3; RO +4.4 → +1.9. (3) *The
  smoking gun*: at its own stations at 1 h the model (0.934) equals the anomaly-persistence kernel
  (0.934) while the GP (a per-station temporal GP on recent history) gets 0.737 and raw persistence
  0.971. The model is not extracting the short-lead value of a station's own recent trajectory; the
  0–4 h gap seen since v1 is a *temporal* deficiency, not spatial, and it is the largest known
  loss. Consequences: (a) report skill against irimap_prod (or fix PyIRI's driver bias, since the
  anomaly target inherits it); (b) top model item = short-lead use of own-station history: unpooled
  tokens for the last ~2 h, recency weighting, or an explicit per-station latest-value/trend
  token; verify with the own-station 1–3 h curve vs the GP; (c) the source-value table stays
  referenced to the indices-only mode.
- 2026-09-09: **PyIRI driver: not a plumbing bug, a wrong index for this cycle.** Oct–Dec 2025 drivers
  handed to PyIRI: trailing-81-day observed F10.7 = 150–155 (→ R12 ≈ 105–111, IG12 ≈ 115–120);
  trailing-365 158–166; SILSO SSN13 128–136 → F10.7 172–179. Production's eSSN (fit to the
  ionosonde network) implies F10.7 ≈ 113–123. So the observed ionosphere in late 2025 behaves like
  an index ~25% below the measured one; IRI driven by any real index over-predicts foF2 (+0.7 MHz
  at stations, +0.2 at RO), and irimap_prod is unbiased only because eSSN is observation-fitted.
  This is the known cycle-25 IRI over-prediction. The anomaly target (obs − PyIRI) therefore
  carries a large slowly varying component tied to F10.7, which is what the indices-only mode
  learns. Actions: (1) replay `iri_essn` for Oct–Dec (running) and report skill against it as the
  honest climatology reference (eSSN at T is available to the live system); (2) consider eSSN (or
  a learned index) as the anomaly *baseline* driver in a later build, so the model spends capacity
  on weather rather than on IRI's index response.
- 2026-09-09: **eSSN as a baseline: still rejected, now with numbers.** Oct–Dec 2025 bias / RMSE by |lat|
  band (stations; RO): PyIRI f107_81 +0.35/1.72, +0.77/1.38, +0.85/1.34 (RO +0.01/1.77,
  +0.41/1.52, +0.58/1.30); IRI-eSSN −0.58/1.86, −0.02/1.20, −0.07/1.09 (RO −0.90/2.05,
  −0.51/1.64, −0.08/1.14); model indices-only −0.02/1.48, +0.01/0.99, +0.21/0.96 (RO +0.04/1.34,
  −0.11/1.28, +0.28/1.08). eSSN fixes the mid/high-latitude bias and makes the equatorial region
  worse than PyIRI (under-prediction 0.6–0.9 MHz, where most RO data is), as Andrew said at
  planning time. The learned climatology is near-unbiased in every band and beats both IRI
  variants everywhere, including equatorial RO (1.34 vs 1.77 / 2.05). That is the strongest
  evidence the indices-only skill is real: it is a latitude-dependent effective-index correction,
  which no single global index can provide. Reporting reference for climatology stays the
  model's own indices-only mode; eSSN is reported only as a midlatitude reference.
- 2026-09-09: **RO skill is inflated by the query kind flag.** Scoring v7p6m_di2 at RO points with the RO
  flag cleared (as a map pixel; `RO_AS_MAP=1 predict.py`): +13.0% over IRI [+12.0, +13.9] vs +23.8%
  with the flag. About 11 points of every RO number reported so far is a learned RO-vs-ionosonde
  systematic offset (RO ionPrf foF2 differs from ionosonde foF2 in a way the model can predict from
  geometry/local time), not map skill. For map quality the map-pixel number is the honest one;
  the model as a map still beats IRI by 13% and the GP by far more at RO (GP −8%, kernel +0.6%).
  Dashboard headline RO tile changed to the map-pixel number with the flagged number in the
  caption; source-table RO columns keep the flag (internally comparable) with a note. Open
  question for later: whether RO truth should be bias-corrected toward ionosonde foF2 before
  scoring, which would make the two numbers meet.
- 2026-09-09: **all RO numbers on the dashboard now scored as map pixels** (RO flag cleared for every source
  combination, the distance strata and the lead curve). v7p6m_di2 at RO over IRI: everything
  +13.0, no spots +12.3, ionosondes only +11.9, no ionosondes +11.9, spots only +11.6, GloTEC only
  +8.5, indices only +10.2; over indices-only: +3.0 / +2.3 / +1.9 / +1.9 / +1.5 / −2.0. The
  observation-attributable part at RO is unchanged by the correction (~3 points); what changed is
  the climatology share (10 of 13 points) and the far-field decay, which the flagged numbers were
  masking: RO error by distance to the nearest ionosonde is now 1.23 → 1.54 MHz (+25%) vs 1.15 →
  1.34 (+16%) flagged. Reported RO skill history (v1..v7p6, +22–24% over IRI) should be read as
  ~+11–13% map skill plus ~+11 points of RO-offset knowledge.
- 2026-09-09: **v8p6 `--val-modes` log through epoch 6 (val held-out foF2): all 1.044, iono-only 1.05,
  spots-only 1.09–1.10, GloTEC-only 1.40→1.105, none 1.09–1.10.** So on validation, spots-only and
  GloTEC-only converge to *at best* the indices-only level: spots add nothing at held-out stations
  alone (consistent with the Oct–Dec paired −0.5 / +1.5-map at RO), GloTEC-only is consistently
  ~1.5% worse than no inputs and not noisy. Cause of the GloTEC lag found: GloTEC exists only from
  2025-05-12, i.e. 1,136 of 17,419 training samples (6.5%), so the GloTEC-only mode is trained on
  ~0.2·0.065·0.3 ≈ 0.4% of samples (~70/epoch) vs ~4% for spots-only; the val column is still
  falling at epoch 6 where the others have flattened. Added `train.py --p-drop-iono-glotec`
  (ionosonde drop rate on GloTEC-era samples, e.g. 0.5 → ~3× the GloTEC-only steps) for the next
  run. Whether GloTEC ever gets *above* the learned climatology is then the honest test of the
  pathway; Phase 3's −33% vs IRI at stations was against a biased IRI, not against this baseline.
- 2026-09-09: **source-dropout curriculum (Andrew's idea): start with mostly single-source samples, ramp to
  the multi-source mix.** Built as annealed whole-source drop rates: `train.py --curriculum N`
  anneals (iono, GloTEC, spots, iono|GloTEC) linearly from `--curriculum-start` (default
  0.6,0.7,0.7) to the run's targets over N epochs. At the start P(all three present) is 3.6% and
  single-source samples dominate, so each pathway must learn to predict on its own before the
  model learns to combine; at the targets (0.2/0.3/0.3) it is 39%. Rationale vs the flat rates: a
  pathway that was never load-bearing early is easy to ignore later (the GloTEC symptom), and the
  fallbacks then lag the primary. Verified the per-epoch rates reach the DataLoader workers.
  Run to try (after v8p6): v8p6 recipe + `--curriculum 6 --p-drop-iono-glotec 0.5 --val-modes`;
  judge by whether the GloTEC-only and spots-only columns rise *above* "none" and hold, and
  whether the all-sources column still reaches ~1.03.
- 2026-09-09: **v8p6 (5-min spot bins ending at T−15 min, otherwise the v7p6m_di2 recipe): best EMA val
  1.043 at epoch 7 vs v7p6m_di2's 1.032 on the same held-out rows; iono-only 1.052, spots-only
  1.099, GloTEC-only 1.114, none 1.097.** Fresher spot bins did not help and cost ~1% on val.
  Candidate reasons: the baseline keyed by the UT hour of the bin midpoint (a 15–45 min mismatch
  against clock-hour climatology), SNR mean-of-spots vs the hourly median, or simply that spot
  tokens carry too little signal beside ionosondes for freshness to matter (spots-only ≈ none in
  both runs). Decision: hourly bins (v7p6 format, `SPOT_RES=60`) stay the production spot format;
  the 5-min pipeline stays available. Paired Oct–Dec check when the checkpoint is pulled down.
- 2026-09-10: **v7p6_cur (di2 recipe + `--curriculum 6 --p-drop-iono-glotec 0.5`) through epoch 8, val
  held-out foF2 (EMA): all 1.048 (di2 1.032 at its epoch 8), iono-only 1.041, spots-only
  1.09–1.11, GloTEC-only 1.12, none 1.09→1.125.** Readings: (1) the curriculum did make spots
  load-bearing on their own: spots-only sits ~2% *above* "none" (equal in every previous run),
  which is the effect it was designed for; (2) the primary is ~1.5% behind di2 at the same
  epoch and still improving slowly, so the curriculum costs best-case skill at least until late
  in the schedule; (3) **GloTEC-only equals or trails "none" with 3× the steps, qf>0 only, and a
  curriculum**: as tokenised (foF2-from-NmF2 anomaly vs PyIRI at 4 lags, log TEC, quality flag)
  GloTEC adds nothing over the learned climatology at held-out stations, and its Phase-3 value was
  against the biased IRI; (4) "none" and spots-only peak around epoch 3 and drift worse as the
  model specialises to the multi-source mix — the modes still converge at different rates, the
  curriculum only moved *when*. Decisions: GloTEC off in the next recipe (`--glotec` omitted;
  pathway and service flag kept, data-side rethink later: TEC itself, or GloTEC's own bias, rather
  than an IRI-referenced foF2 anomaly); curriculum kept as an option for the spot pathway, not the
  default; next model work is the fork's top item, the 0–4 h own-station temporal deficiency.
- 2026-09-10: **v7p6_cur_4ng (no GloTEC; Muon + EMA + `--p-drop-iono 0.1 --dropout 0.1 --curriculum 4`;
  best EMA epoch 6, val 1.038), Oct–Dec 2025, paired, foF2.** vs v7p6m_di2: holdout −0.6%
  [−1.4, +0.3], full −1.1% [−2.1, −0.1], RO (flagged) +0.1%; = v7p6 at stations; vs GP full +6.9%,
  holdout +7.4%; σ 65/93% (di2 63/92). RO as map pixels +13.5% over IRI (di2 +13.0). Source value
  at held-out stations over its own indices-only mode: everything +6.2%, ionosondes-only +6.4%,
  **spots-only +2.2% [+0.9, +3.4]** (di2: −0.5%): the first checkpoint where spots alone beat the
  learned climatology at stations with a CI clear of zero. Reading: the milder curriculum
  (0.1/0.1/4) bought a load-bearing spot pathway and better calibration for ~1 point of
  best-case skill at stations; RO unchanged. Neither dominates di2; both beat production by the
  same margin. At RO as map pixels, over its own indices-only mode: everything +3.1%, ionosondes-only +3.2%,
  **spots-only +2.6% [+2.0, +3.3]** (di2 +1.0%): spots alone recover ~85% of what the full
  ionosonde network gives at RO, vs a third for di2. Recipe choice is a preference: cur_4ng if the
  spots-only outage mode matters (production has the loader), di2 if only the best case does.
- 2026-09-10: dashboard republished from **v7p6_cur_4ng** (same URL; RO charts and columns now labelled
  as map-pixel scored; four-row source table, GloTEC gone). **W3USR standalone with cur_4ng**
  (`w3usr_cpp6`, May–Jun 2025, own baseline, receiver spots only, no ionosondes): skill over IRI at
  stations +43.7% vs the model's own no-inputs +43.9% and full inputs +48.4%; at RO +31.2% vs
  no-inputs +36.3% and full +37.7%. RO error by distance from the receiver, receiver-only /
  no-inputs / full: <1500 km 1.121 / 1.205 / 1.052; 1500–3000 1.161 / 1.202 / 1.089; 3000–6000
  1.221 / 1.177 / 1.126; >6000 1.317 / 1.205 / 1.187. Same shape as with v7p6 but sharper: a lone
  receiver is worth 7% within 1500 km and 3% to 3000 km over the learned climatology, and is
  *harmful* beyond 3000 km (its far control points are sparse, one-sided tokens the model never
  saw alone). Radius test (`predict.py --spot-within`): tokens within 3000 km → RO by distance 1.101 /
  1.146 / 1.201 / 1.293 (better everywhere than unrestricted, still worse than no-inputs beyond
  3000 km); within 1500 km → worse everywhere (1.123 / 1.176 / 1.255 / 1.351). So the far-field
  harm is not the far tokens: a handful of near-receiver tokens shifts the *global* answer, because
  attention is global and training never showed a spatially isolated token set (the network is
  always worldwide). Standalone options: (a) inference blend, receiver model within ~3000 km,
  climatology beyond; (b) the principled fix, *spatial* spot dropout in training (with some
  probability keep only tokens inside a random-centred cap of random radius), so the model learns
  that absence outside a region is not information — one more training knob, untested.
  **RO by mission** (Oct–Dec, map pixels, cur_4ng): COSMIC-2 +14.1% over IRI (1.464 vs 1.705),
  PlanetiQ +10.6% (1.305 vs 1.460); the truth set is 79% COSMIC-2 / 21% PlanetiQ in the window
  (dashboard caption corrected from "COSMIC-2 profiles").
- 2026-09-10: **close-in (0–4 h own-station) work started.** Diagnosis (fork, 2026-09-09): at its own
  stations at 1 h the model equals anomaly persistence (0.934) while the GP gets 0.737; the model
  cannot cleanly find and read its own station's trajectory among ~1k tokens, pooling hides the
  last hour's shape, and short leads are 1/24 of the loss. Built: (1) **nearest-station state on
  the query** (`build_samples.py --qstate`, `query_state()`): 8 features — present, dist/1000 km,
  age/24 h, the nearest input station's latest normalised anomaly (foF2, hmF2, MUF), and its foF2
  anomaly change over the last 1 h and 3 h — computed from the same pooled rows the tokens come
  from; present=1 within 1000 km with data in the last 3 h (own stations at dist 0, most held-out
  stations via a neighbour, 3% of RO), zeros otherwise. F_QRY 10→18, recorded in samples and
  checkpoints (`f_qry`); service and `forecast.py` compute it the same way at forecast time.
  Verified on one sample: own stations present 0.94 at dist 0, age 0.64 h, first 10 query columns
  bit-identical to v7p6. (2) **`train.py --short-lead-weight W --short-lead-h H`**: loss multiplier
  for queries with lead ≤ H (Andrew: give it the tool and the encouragement; 3 / 3 h is the first
  try). (3, not built) unpooled tokens for the last 2 h. Samples `train_v9`/`val_v9` (v7p6 recipe
  + `--qstate`) building. Run: cur_4ng recipe on v9 with `--short-lead-weight 3`; judge on the
  own-station 1–3 h full-mode curve vs the GP (0.737 at 1 h) and the usual paired report.
- 2026-09-10: **spatial skill maps at RO** (`/kass/forecast/maps/skillmap/ro_skill_cells.png`, artifact "Where
  Spots Help"): 10° cells with ≥300 profiles (593 cells), Oct–Dec 2025, cur_4ng, map pixels.
  Spots-only vs indices-only: median +2.0%, 296 cells >+2%, 130 <−2%; positive over the Americas,
  N Atlantic, Europe, Japan, S Indian Ocean; negative over the SE Pacific and south of
  Australia/NZ. **The all-inputs map shows the same negative blocks**, so they are not a spot
  property: in regions with no observations of any kind, any tokens shift the global answer away
  from climatology and make it worse — the single-receiver far-field harm, seen in the network
  case. Indices-only vs IRI is worse than IRI over India/SE Asia and the SE Pacific (no ionosondes
  to learn from). Consequence: the spatial-dropout / "absence is not information" training idea is
  a network issue, not just a standalone one; a distance-to-nearest-observation gate at inference
  would be the cheap mitigation.
- 2026-09-10: v9 samples ready (17,419 / 736; state present at 99% of own-station and 49% of held-out
  queries). **Leak closed (Andrew's catch):** the query-state columns are ionosonde data, so they
  are now zeroed wherever the ionosonde tokens are dropped — `Samples` (`--p-drop-iono`),
  `evaluate(drop=("iono",…))` (the `--val-modes` spots-only / none columns) and `predict.py
  --drop-iono`. Verified: dropped samples have all-zero state columns with the first 10 intact,
  and `evaluate(drop=iono)` equals evaluation on hand-zeroed columns. The service never drops
  ionosondes, so it is unaffected. The v9 run can start.
- 2026-09-11: **v9 (cur_4ng recipe + `--qstate` + `--short-lead-weight 3`), Oct–Dec 2025, paired, foF2: the
  0–4 h gap is closed and reversed.** Own-station full mode by lead, v9 / GP / cur_4ng: 1 h 0.655 /
  0.737 / 0.992; 2 h 0.779 / 0.908 / 1.003; 3 h 0.850 / 0.979 / 1.005; 4 h 0.906 / 1.008 / 1.009;
  6 h 0.964 / 1.060 / 1.016. vs cur_4ng: full +4.0% [+2.9, +5.0], holdout +0.4% [−0.6, +1.2], RO
  (flagged) −0.6% [−0.9, −0.3]; MUF full +2.8%; vs GP full **+11.0%** [+8.7, +13.1] (cur_4ng +6.9),
  holdout +6.7%; σ 67/94% (best yet). Val 1.034 at epoch 5; val-modes iono 1.030, spots 1.075,
  none 1.091. So the diagnosis held: handing the decoder the station's own latest anomaly and
  trend (plus tripling the short-lead loss weight) turns persistence-level short-lead skill into
  GP-beating skill at every lead, while held-out and RO skill are untouched (the feature fires at
  held-out stations via a neighbour and does nothing there, as expected). Outage views: skill over IRI
  holdout / RO-map — everything +28.8 / +14.9, ionosondes-only +29.2 / +15.4, spots-only +25.9 /
  +13.8, indices-only +24.9 / +13.2; over indices-only: +5.2 / +2.0, +5.6 / +2.5, spots-only +1.5
  / +0.7. RO as map pixels +14.9% over IRI (cur_4ng +13.5); the indices-only mode also improved
  (+13.2 vs +10.7 at RO). Spots' marginal value shrank as the climatology improved (+1.5 / +0.7
  vs +2.2 / +2.6). Dashboard republished from v9. **v9 is the production candidate.** Andrew: no cur_4ng experiment; the live experiments go from
  v7p6 to v9 (`2026-09-v9`, `2026-09-v9-spots` added to `deploy/scheduler.patch.md`; v9 has no
  GloTEC pathway, `glotec => 0`). Live re-score after a month compares v4, v7p6 and v9 against
  production on the same runs.
- 2026-09-11: **W3USR standalone with v9** (`w3usr_v9` samples rebuilt with `--qstate`; receiver spots only,
  no ionosondes, own baseline, May–Jun 2025). Skill over IRI at stations +42.9% (r3000 +43.3%) vs
  v9's own no-inputs +44.4% and full +48.2%; at RO +31.9% (r3000 +32.4%) vs no-inputs +35.8% and
  full +37.4%. RO by distance from the receiver, receiver-only / within-3000 km / no-inputs:
  <1500 km 1.133 / 1.102 / 1.146; 1500–3000 1.179 / 1.169 / 1.167; 3000–6000 1.225 / 1.220 /
  1.173; >6000 1.300 / 1.289 / 1.219. Stations <1500 km 0.770 / 0.760 / 0.771. So with v9 the
  lone receiver is worth ~1–4% inside 1500 km only (with the 3000 km token gate), nothing at
  1500–3000 km, and it still harms the far field. The learned climatology improved faster than
  the receiver's marginal value; the standalone product needs the spatial-dropout training (or an
  inference blend) before it is worth shipping, and even then the ceiling looks like a few
  percent inside the single-hop footprint. Back-burner stands.
- 2026-09-11: **v9 fails coherence: discs around every station** (`out/fof2_lead00.png`, Andrew). Cause: the
  query-state feature switched on at a hard 1000 km radius with nearest-station-wins, so the map
  steps at every disc edge and at the Voronoi boundaries between neighbours; the model cannot make
  a smooth field from a discontinuous input. Fix in `query_state`: Gaussian kernel weights
  w = exp(−(d/500 km)²) over all stations within 1500 km, state columns = weight-normalised mean
  scaled by min(Σw, 1) so every column tapers to zero, first column = min(Σw, 1), distance column
  = nearest/1500 (1 = none in reach). Verified: largest step between adjacent 8.5 km queries 0.015
  (was 1.5), smooth blend between two stations 600 km apart, station itself still gets its own
  state exactly (w = 1). Building `train_v9b`/`val_v9b`; v9 must be retrained on them (v9b) and
  is not a production candidate until the maps pass the gate. The skill numbers stand as
  evidence for the mechanism, not for the checkpoint.
- 2026-09-11: **v9b training log** (continuous query state, otherwise v9 recipe): EMA val held-out foF2
  1.124 → 1.030 (ep 5) → 1.010 (ep 10), the best held-out val ever by 2% (v9 1.034, di2 1.032): the
  continuous feature reaches held-out stations through their neighbours where the hard cut mostly
  did not. Single-source modes: spots-only peaks at 1.061 (ep 5) and drifts to 1.08, none 1.08→1.11,
  the usual rate mismatch; at ep 10 spots-only is still +2.5% above none. Trade accepted: the
  primary is the production metric. Added `train.py --keep-epochs` (EMA checkpoint per epoch) so
  a later pass can pick per mode or average epochs 5–10 without retraining.
- 2026-09-11: **v9b (continuous query state), Oct–Dec 2025, paired, foF2.** Maps: the discs are gone
  (`/kass/forecast/maps/v9b_check`); Andrew: near-station gradients are still steeper than the GP's
  (e.g. 45N 70W, 35S 150E at 0 h) — the state kernel's 500 km footprint, with the model trusting a
  station's own anomaly fully at d=0. vs cur_4ng: holdout **+2.7%** [+1.7, +3.7] (v9 +0.4), full
  +5.7% [+4.7, +6.9], RO flagged +1.0%; vs GP full **+12.6%** [+10.3, +14.6], holdout **+9.0%**
  [+5.3, +12.6]. Own-station by lead 1..6 h: 0.630 / 0.761 / 0.833 / 0.890 / 0.924 / 0.944 (GP
  0.737 / 0.908 / 0.979 / 1.008 / 1.039 / 1.060); held-out 1..6 h 0.946–0.957 (v9 0.959–0.982). RO
  as map pixels +14.9% over IRI; RO map by distance to the nearest ionosonde 1.205 / 1.222 / 1.256
  / 1.345 / 1.431 / 1.477 (v9 1.214…1.501, cur_4ng 1.240…1.528, GP 1.392…2.051): **the steep
  near-station structure scores better, not worse, at RO points within 250 km** (1.205 vs 1.240
  for the disc-free cur_4ng and 1.392 for the smooth GP), so the bumps carry real local signal.
  Source value at holdout / RO-map over IRI: everything +30.4 / +14.9, ionosondes-only +30.4 /
  +14.8, spots-only +25.6 / +13.9, indices-only +23.6 / +12.5; over indices-only: +8.8 / +2.7,
  +8.7 / +2.6, spots-only +2.4 / +1.6. Coherence at 2025-09-09 00Z (9 leads at 3 h), v9b / GP: anomaly RMS 1.10 / 1.02,
  fine-scale 0.0003 / 0.0000, grad_far 0.054 / 0.025 (2.2×), grad_near 0.063 / 0.021 (3.0×),
  change per hour 0.21 / 0.11 (geo), sign flips 5.1% / 0.0%. Passes the provisional gate on this
  day (fine-scale < 0.01, grad_far ≤ 3.5× GP, sign flips < 10%); the near-station gradient is 3×
  the GP's, which is the structure Andrew flagged, and the RO bins say it scores. Month-long gate
  run (October 2025 daily 00Z, v9b vs the existing GP grids) launched. Production candidate,
  subject to the gate.
- 2026-09-11: **v9b passes the monthly coherence gate** (October 2025 daily 00Z, 31 days, vs the GP and v5 on
  the same days; `eval/coherence_oct2025_v9b.parquet`). Monthly means GP / v5 / v9b: anomaly RMS
  0.75 / 1.07 / 1.00; fine-scale 0.000 / 0.0007 / 0.0003; grad_far 0.020 / 0.061 / 0.055 (v9b
  2.7× GP, gate 3.5×); grad_near 0.022 / 0.074 / 0.077 (3.5×, not gated); change per hour 0.12 /
  0.26 / 0.23; sign flips 2.4% / 9.0% / 8.8% (worst day 14.5%). So v9b is *smoother* than v5 on
  every gated metric while adding the near-station structure; the near-station gradient is the
  one place it exceeds v5, by 4%. **v9b is cleared for production**: promote once the live
  experiment (`2026-09-v9b`) confirms; scheduler notes updated to v9b. Andrew, on the near-station
  structure: "still not realistic, but if it's the best approximation to reality we can get we
  live with it." Accepted with that reservation; if map realism becomes a product requirement,
  the levers are a two-scale query state (separating the station-local part from the regional
  part) or a smoothness penalty on grid queries in training, both untested.
- 2026-09-11: **split encoding, zero-training test** (Andrew's question: encode sources in separate encoder
  passes, concatenate the memories, decode once; `~/.claude/jobs/812d1d6e/tmp/split_encode.py`,
  v9b, every 2nd val_v9b sample). foF2 RMSE held-out / own / RO: joint 1.007 / 0.972 / 1.247;
  split 1.056 / 1.015 / 1.271; split with the second pass's null+global dropped 1.056 / 1.015 /
  1.271; ionosondes-only 1.009 / 0.970 / 1.250. So the concatenated-memory interface works
  mechanically, but a spot memory encoded without ionosonde context is misread by a decoder
  trained on joint memories: 5% worse at stations, 2% at RO, and worse than dropping spots
  entirely. Duplicated null/global tokens are irrelevant. Verdict: viable only as a trained
  architecture (per-source encoders + concatenated memory, optionally one shared mixing layer),
  not as a drop-in; the payoffs would be O(Σn_i²) encoder cost, incremental re-encoding of the
  source that changed, sources encoded in different places, and distance-masked decoder attention
  for locality. Not scheduled.
- 2026-09-16: **GloTEC / spots: where the value is, and the plan to extract it (Andrew: a genuine gain, or
  at least "all sources don't harm").** Stratifying v9b's map-pixel RO skill by distance to the
  nearest ionosonde: spots-only over indices-only is 0.0 / 0.0 / +0.1 / −0.3 / **+1.6 / +3.4**% for
  <250 / 250–500 / 500–1000 / 1000–2000 / 2000–4000 / >4000 km, and with ionosondes present spots
  still add +0.6% beyond 4000 km (1.477 vs 1.486). So the spot pathway does what it should — it
  helps where there is nothing else — and the "null" verdicts were the 2.7M-row global average
  diluting 600k far-field rows. Reading of Andrew's two instincts: (1) the cross-attention
  pathway is probably not the binding constraint: the joint encoder already lets sources interact,
  and the redundancy at scored points near stations is real; the missing piece is behaviour where
  observations are sparse (the negative RO cells) and the far-field harm, which are what the
  spatial-dropout idea targets; (2) memorisation is binding: train loss → 0 (negative) by epoch
  8–10 because 8 samples/day share 21 of 24 h of the same station targets and the NLL rewards
  σ→0 on memorised points, so single-source modes and cross-source interactions never get the
  late epochs they need. Built now (no rebuild): `train.py --target-noise 0.15` (Gaussian noise on
  the normalised targets at the autoscaling-error level, so exact fits are impossible) and
  `--logvar-floor -4` (σ cannot collapse below ~0.2 MHz in the loss). Plan: (A) v9b recipe +
  target noise + logvar floor, 20 epochs, `--keep-epochs --val-modes`: does val keep improving past
  epoch 10, and do the single-source columns stop decaying? (B) spatial dropout of observation
  tokens at load time (keep only tokens inside a random cap with some probability; recompute the
  query state from the kept rows) so absence is not information — targets the negative RO cells
  and is the precondition for GloTEC/spots showing value in oceans; (C) report source value by
  distance-to-ionosonde as standard, since the global average hides it; (D) architecture (per-source
  encoders, gated fusion) only if A–C leave the far field on the table. GloTEC after the qf>0
  filter still trailed climatology alone (v8p6 val), so its next test rides on A+B.
- 2026-09-16: dashboard republished from **v9b** (same artifact, now at https://claude.ai/artifact/6Dg2Xsm7dNjBFk23shMzuN
  after a link-format change): +9.0% / +12.6% over the GP at held-out / all stations, +14.9% over
  IRI at RO as map pixels, σ 63/91%; source table everything +30.4 / +14.9, ionosondes-only +30.4 /
  +14.8, spots-only +25.6 / +13.9, indices-only +23.6 / +12.5 over IRI (holdout / RO map), and
  +8.8 / +2.7, +8.7 / +2.6, +2.4 / +1.6 over indices-only; the far-field note on spots is in the
  source blurb.
- 2026-09-16: **v9b3 (v9b + `--glotec` + target noise 0.15 + logvar floor −4, 20 epochs): early epochs
  unhurt, memorisation unchanged.** EMA val 1.030 best at epoch 7, then 1.046 (10), 1.070 (14),
  1.088 (20); train loss still → 0.003; single-source columns decay from epoch 5–7 as before.
  Also 2% worse than v9b at its best (1.010), confounded by GloTEC being back on. Why target noise
  cannot work here: the same clean target appears in ~8 samples per epoch with independent noise
  each time, so the clean value is recoverable by averaging and gets memorised anyway; noise
  removes only the incentive to fit the noise (the loss floor moved, the curve did not). Built
  instead: `train.py --lead-window 3` — in training each sample keeps only the targets inside one
  random 3-h lead window per epoch, so each target hour is seen ~once per epoch instead of ~8×,
  with the lead distribution over an epoch unchanged (verified: 6 draws cover different windows,
  60-draw lead histogram flat). Next run: v9b recipe (no GloTEC) + `--lead-window 3 --keep-epochs
  --val-modes --epochs 20`, noise and floor off; the question is whether the val curve flattens
  past epoch 10.
- 2026-09-17: **v9b4 (v9b + `--glotec` + `--lead-window 3` + target noise 0.15 + logvar floor −4, 20
  epochs; Andrew: the noise/floor were left on by mistake): memorisation reduced, no gain from
  the extra epochs.** Train loss ends at 0.072 (v9b3 0.003; v9b ≈ 0 by epoch 10) and the val
  drift after the best epoch is halved (1.037 at 11 → 1.057 at 20, vs 1.030 → 1.088 for v9b3), so the
  window does what it was built for. But the plateau is flat, not rising: epochs 8–20 sit at
  1.037–1.057 with the single-source columns flat too. Reading: with the repeated-target incentive
  removed, 1.3M params has nothing further to learn from these 17k samples — the limit is capacity
  / information, not epochs. Confounds to clear first: both v9b3 and v9b4 ran with `--glotec` on and
  both sit ~2.5% above v9b (1.010, no GloTEC), and v9b4 also carried the noise/floor, so the
  window's own effect on the primary is unmeasured. Next runs: (1) v9b recipe + `--lead-window 3`
  only, **no** `--glotec`, no noise/floor, 12 epochs — clean read of the window against 1.010; (2) if the window is neutral-or-better, the long-deferred capacity run on top of it
  (`--d 256 --layers 6`, ~5M params, ~4× step time), which is now safe to run for 15+ epochs.
- 2026-09-17: **v9b5 (v9b recipe + `--lead-window 3` only, no GloTEC, 12 epochs): the window costs 3%.**
  Best EMA 1.041 at epoch 8 vs v9b 1.010; plateau 1.041–1.047 from epoch 6; spots-only 1.09, none
  1.12 (v9b 1.08 / 1.10). Train loss 0.058 at the end (v9b ≈ 0), so it does suppress memorisation,
  but the optimum it reaches is worse: with max_q 1024 the number of targets per sample is the
  same either way, so the window does not reduce exposure, it reduces the *diversity* of targets
  within a step (one lead band), which is a noisier, worse gradient. Dropped. Scorecard for the
  "train longer" instinct: dropout 0.2 (+3 usable epochs, no gain past 8), target noise + logvar
  floor (no change), lead window (−3%). The train-loss→0 memorisation is real but is not what
  caps validation; v9b's own post-peak drift was only 1.010→1.015 over epochs 10–12. Conclusion:
  at 1.3M params the model is information/capacity-limited around epoch 8–10 on these samples;
  the remaining lever for cross-source learning is capacity with early stopping as usual. Next:
  v9b recipe (no window, no noise, no GloTEC) with `--d 256 --layers 6`, 12 epochs,
  `--keep-epochs --val-modes`; judge on held-out val vs 1.010 and on the spots-only column.
- 2026-09-17: **capacity run v9b_d256 (d 256, 6 layers, 6.7M params, v9b recipe): worse, and it memorises
  faster.** Best EMA 1.028 at epochs 4–6 (v9b 1.010 at 1.3M), then train loss goes negative by
  epoch 6 and val climbs to 1.077 by epoch 11; spots-only never below 1.076, none 1.10→1.13. So the
  limit is the information in 17k samples × ~40 stations, not capacity: a 5× bigger model finds
  no more structure and overfits sooner. Together with the epoch experiments this closes the
  "learn cross-source dependencies with more model/more epochs" line at the current data size.
  Where a genuine multi-source gain exists is already measured: spots beyond 2000 km from any
  ionosonde (+1.6 / +3.4% over climatology at RO). Remaining route for "all sources don't harm":
  spatial dropout of observation tokens in training (absence is not information) plus a
  distance-to-nearest-observation gate at inference; and per-source encoders (the fork's topic)
  only as an architecture experiment, not a capacity one. v9b stands as the production candidate.
- 2026-09-17: **run-to-run noise (Andrew): v9b's 1.010 is not reproducible; 1.03–1.04 is typical for the
  recipe.** Sample loading was unseeded. This reframes the last week's single-run comparisons:
  differences under ~2% on held-out val (window −3%, capacity +2%, GloTEC-on +2.5%, noise ±0) are
  within one or two draws of noise and are *not established*; only the v9/v9b structural gains
  (own-station 1 h 0.63–0.66 vs 0.99, the RO-by-distance shifts) are large enough to stand. Added
  `train.py --seed N` (torch, numpy, DataLoader generator, per-worker per-epoch numpy streams; every
  rng in the sampler now derives from it) — verified identical batches across two seeded runs.
  From now on a recipe change needs two seeds, and the noise floor comes from two seeds of the
  v9b recipe. **Spatial dropout built**: `--p-spatial 0.3 --spatial-km 1500,6000` keeps only the
  observation tokens (all kinds) inside a random cap centred on a random token, and recomputes
  the query-state columns from the kept ionosonde rows (verified: capped draws keep 10–30% of
  tokens, station-state coverage drops from 0.67 to 0.1–0.2 of queries, weight at kept stations
  still 1.0). Runs (Andrew): v9b recipe `--seed 1` and `--seed 2` (noise floor), then the same with
  `--p-spatial 0.3` at both seeds; judged on the RO skill map's negative cells, the W3USR far field,
  spots value beyond 2000 km, and held-out val within the seed spread.
- 2026-09-18: **noise floor for the v9b recipe: seed 1 → 1.040 (epoch 7), seed 2 → 1.036 (epoch 8).** So
  the recipe's typical held-out val is ~1.038, and the production v9b checkpoint (1.010) is a
  ~3% lucky draw of the same recipe. Consequences: (1) two seeds differ by 0.4%, so the seed
  spread is small and the outlier is v9b, not the seeds — a change is real when both seeds move
  by more than ~1%; (2) v9b as a *checkpoint* is still what it measures (its Oct–Dec numbers are
  its own), but expectations for retrains and fine-tunes should be set at 1.03–1.04, and the
  live experiment is the honest test of the deployed weights; (3) the earlier single-run verdicts
  are re-read against 1.038, not 1.010: window 1.041 (neutral), capacity 1.028 (better by 1%,
  not worse), GloTEC-on 1.030–1.037 (neutral), noise/floor 1.030 (neutral). Paired Oct–Dec,
  seeds 1 / 2 / lucky v9b vs cur_4ng: holdout −0.1 / +0.3 / +2.7%, full +4.7 / +4.9 / +5.7%; vs GP
  full +11.6 / +11.9 / +12.6%; own-station 1 h 0.640 / 0.645 / 0.630 (cur_4ng 0.992, GP 0.737);
  held-out 1–4 h 0.98–0.99 / 0.98–0.99 / 0.94. **So the structural gains are reproducible — the
  own-station short-lead skill and the ~+5% full-mode / ~+12% vs-GP margins — and v9b's held-out
  +2.7% over cur_4ng was the lucky part** (typical draw: held-out equal to cur_4ng). The deployed
  v9b keeps its own numbers; the recipe's expectation for retrains is "cur_4ng at held-out
  stations, +5% at own stations, GP-beating at every lead".
- 2026-09-19: **v9b + `--p-spatial 0.3`, seeds 1 and 2: held-out val ~1.05 for both** (recipe floor 1.036–1.040),
  so the cap costs ~1.2% on the primary and the loss is real (both seeds, > 1%). What it buys (seed 1
  scored; Oct–Dec unless noted): **RO skill-map negative cells halved** — all-inputs vs own
  climatology <−2% in 57 of 593 cells (v9b 110), spots-only 53 (114), worst cell −11.5% (−13.7);
  **spots-only now above climatology in every distance bin** (+1.2…+2.7%; v9b 0 inside 2000 km) and
  nearly equal to ionosondes-only at RO (+13.0 vs +13.3 over IRI); **the W3USR far-field harm is
  gone**: receiver-only RO by distance 1.119 / 1.158 / 1.151 / 1.207 vs its own no-inputs 1.160 /
  1.173 / 1.165 / 1.210 (v9b: 1.133 / 1.179 / 1.225 / 1.300 vs 1.146 / 1.167 / 1.173 / 1.219),
  i.e. a lone receiver now helps at every range and never hurts; overall +1.4% at stations and
  +0.4% at RO over climatology (v9b −1.4 / −6.2). Cost: paired vs cur_4ng, holdout −1.3 / −1.2%
  (seeds 1 / 2; unmodified seed −0.1), full +3.1 / +3.4% (+4.7); RO map +13.3% over IRI vs v9b's
  +14.9 (lucky; seeded baseline pending), with the indices-only mode itself weaker (+11.0 vs
  +12.5). Reading: the cap does exactly what it was built for — absence is no longer information —
  and it costs ~1.2–1.5 points of best-case skill, part of it through a weaker learned
  climatology (30% of samples now train it on partial worlds). Knobs if the cost matters: lower
  p_spatial (0.15), or wider caps (3000–8000 km). Decision (Andrew): production trades best case
  for robustness or not.
- 2026-09-19: **v9b_spatial_3 (Andrew: `--p-spatial 0.25 --curriculum 3 --logvar-floor -4 --target-noise
  0.05`, val 1.042) keeps the far-field gains and gives back most of the cost.** Paired vs
  cur_4ng: holdout −0.4% [−1.5, +0.5] (spatial_1 −1.3, seed1 −0.1), full +4.3% [+3.2, +5.4]
  (spatial_1 +3.1, seed1 +4.7). RO as map pixels, with the seeded baseline now available:
  spatial_3 +14.4% over IRI vs seed1 +12.9% (spatial_1 +13.3); its indices-only mode +11.8 vs
  seed1's +11.1 — so against a fair baseline the cap *improves* the far field rather than costing
  it; v9b's +14.9 was the lucky draw. Negative cells (all inputs below own climatology): 55 of 593
  (seed1 127, spatial_1 57), worst −12.3% (seed1 −22.9%); spots-only above climatology in 318
  cells, below in 54. W3USR receiver-only: +2.4% at stations, +1.0% at RO over its climatology,
  RO by distance 1.090 / 1.145 / 1.149 / 1.201 vs 1.175 / 1.181 / 1.175 / 1.209 — helps at every
  range, no far-field harm. **Recommendation: spatial_3 replaces v9b as the production candidate**
  (best case within noise of the recipe, robust to sparse inputs, spots load-bearing, standalone
  works); rerun the coherence gate on it before the swap.
- 2026-09-19: **spatial_3 passes the monthly coherence gate, smoother than v9b.** October 2025 daily 00Z,
  GP / v9b / spatial_3: anomaly RMS 0.75 / 1.00 / 0.92; fine-scale 0.000 / 0.0003 / 0.0004; grad_far
  0.020 / 0.055 / 0.048 (2.4× GP); grad_near 0.022 / 0.077 / 0.070; change per hour 0.12 / 0.23 /
  0.22; sign flips 2.4 / 8.8 / 9.7% (worst day 16.9%). The cap makes the far field calmer (less
  gradient, smaller anomalies away from stations) at the cost of slightly more sign flips, still
  under the 10% gate. **v9b_spatial_3 is the production candidate**; scheduler experiments named
  `2026-09-sp3` / `2026-09-sp3-spots` in `deploy/scheduler.patch.md`; dashboard republished from
  spatial_3 (same URL): +6.1% / +11.3% over the GP at held-out / all stations, +14.4% over IRI at
  RO as map pixels, σ 68/94%; source rows (holdout / RO-map over IRI; over indices-only):
  everything +28.3 / +14.4 (+7.4 / +2.9), ionosondes-only +28.3 / +14.3 (+7.3 / +2.8), spots-only
  +25.4 / +13.8 (+3.7 / +2.2), indices-only +22.5 / +11.8.

## Open questions

1. Test set 2025-01 → 2026-06 frozen? Live experiments already ran over it, so you know roughly
   how the old models did there. That contaminates your judgement slightly, not the new models.
   Assuming yes unless you object.
2. Spot data formats: does the source hand us ClickHouse native dumps or raw text? Determines
   whether Phase 0 stands up ClickHouse on scranton or converts straight to parquet.
