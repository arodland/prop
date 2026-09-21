# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`prop` (kc2gprop) is an HF radio propagation prediction system built around real-time and
forecast models of the ionosphere. It ingests ionosonde/GNSS/COSMIC observations, assimilates
them into IRI-2020-based maps, runs ML forecast models, and serves predictions via a web API and
static frontend (`www/`).

It is a collection of independently-built, independently-deployed services (mostly one per
top-level directory) that share a Postgres database and communicate over HTTP within a podman pod.
There is no monorepo build system, package manager, or test runner that spans the whole tree —
each directory is its own project. `cd` into the relevant directory before doing anything
language/framework specific.

## Services (top-level directories)

| Directory | Language | Role |
|---|---|---|
| `api` | Python/Flask | Public HTTP API over the Postgres `measurement`/prediction tables |
| `scheduler` | Perl/Mojolicious + Minion | Cron-like job scheduler; each `Task::*` plugin enqueues work for another service on a 15-minute cadence |
| `assimilate` | Python | Assimilates observations (GloTEC, ionosondes) into maps |
| `irimap` | Python + Fortran (`irimap` binary) | Generates IRI-2020 ionosphere maps |
| `iri2020` | Fortran/Python | Vendored IRI-2020 model, built as its own image and used by other services (`irimap`, `diffusion`) |
| `pred` | Python | Legacy/statistical foF2 prediction (Gaussian process fits over `cs`/delta-cs) |
| `diffusion` | Python/PyTorch | ML forecast model (DiT diffusion transformer) — see below |
| `essn` | Python | Effective sunspot number (eSSN) computation |
| `ipe` | Python | IPE model integration |
| `cosmic` | Python | COSMIC radio-occultation data backfill |
| `holdout-eval` | Python | Model evaluation against held-out observations |
| `history` | Perl | Historical data service |
| `loader` | Perl | Ionosonde (SAO-4/SAOXML) ingestion from NOAA/GIRO/AUS/INGV — has its own `loader/CLAUDE.md` |
| `raytrace` | Python/Cython | HF raytracing |
| `iturhfprop` | Perl + C++ | ITU-R HF propagation prediction wrapper |
| `renderer` | Python | Map/plot rendering |
| `storm` | Python | Geomagnetic storm detection |
| `www` | Static HTML/JS | Frontend, served independently of the above |

Ports for local services are assigned in `etc/ports.env` (5500–5520 range, bound in the podman
pod). DB connection env vars (`DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`) come from
`etc/db.env.in`, consistently across every service.

## Build / deploy

Each service builds as its own container image:

```bash
podman -r build --tag <image-name> <dir>     # per-service, see buildall for the full tag list
./buildall                                    # builds every image via podman
./buildkitall                                 # equivalent, via buildctl/buildkitd with a shared cache
```

Runtime deployment is systemd + podman on the target host: unit files live in `systemd/`, one pod
(`prop-podman-pod.service`) hosts the service containers, and most services also have a paired
`.timer` unit or are triggered by the `scheduler`'s Minion queue. There is no single "run
everything locally" command — check the relevant service's Dockerfile/README for how it expects
to be invoked standalone.

## Database

Postgres is the shared source of truth. Core tables (see `loader/CLAUDE.md` for the fullest
schema description):
- `station` — ionosonde station metadata
- `measurement` — parsed observations (foF2, MUF(D), hmF2, TEC, etc.), one row per station/time
- Prediction/map tables owned by `api`, `assimilate`, `irimap`, `pred` — inspect each service's
  models rather than assuming a shared schema module; there isn't one.

## The `scheduler` pipeline

`scheduler/app/main.pl` runs a Minion job queue on a 15-minute cadence (5 minutes early: `:10,
:25, :40, :55`). Each `scheduler/app/lib/Task/*.pm` plugin enqueues an HTTP call to one other
service (eSSN → Pred → IRIMap → IPE → Assimilate → BandQuality → Render → HoldoutEvaluate →
Cleanup). Read this file first when tracing "what triggers what" across services.

## `diffusion/` — ML forecast model

This is the most actively-developed and most complex piece. Runs in a container (`./run` script,
uv-managed `pyproject.toml`, PyTorch/Lightning/Diffusers on CUDA) — **do not execute training or
inference scripts directly**; the user runs them in the container.

- Model: `app/models_forecast.py` — `ForecastObservationConditionedDiT`, a DiT operating in a
  VAE latent space, cross-attending to a `SpatioTemporalObservationEncoder` over ionosonde/GloTEC
  observations, conditioned on the previous hourly map and global params (SSN, time-of-day, etc).
- VAE: purpose-built `TinyIonoVAE` (not TAESD, despite older docs/comments implying otherwise).
- Dataset generation: `data/generate_forecast_dataset.py`, built from `iri2020`/`irimap` output.
- `app/attic/` holds superseded training/model scripts — don't treat them as current.
- The many top-level `*.md` design docs (`LOCAL_OBS_ATTENTION_DESIGN.md`,
  `FORECAST_README.md`, etc.) and `app/analyze_*.py`/`debug_*.py`/`diagnose_*.py`/`visualize_*.py`
  scripts are working notes and one-off investigation tools, not maintained references — treat
  them as historical context, verify claims against current code before relying on them.
- Deeper architecture/experiment notes for this model already live in the auto-memory system
  (VAE latent sizing, observation tensor layout, map channel normalization, dataset generation
  defaults) — check there before re-deriving them from scratch.

## Conventions across services

- Python services: Flask app served by `uwsgi` in production (see each `Dockerfile`'s `CMD`),
  Postgres via `psycopg`, HDF5 I/O via `h5py`+`hdf5plugin` where present.
- Perl services: `Mojolicious`/`Mojolicious::Lite`, `Mojo::Pg` for DB access.
- No shared linter/formatter config exists across the repo; match the style already in the file
  you're editing.
