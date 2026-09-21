"""GloTEC tokens for an issue time (PLAN.md Phase 4b): one token per sampled 2.5° cell per lag, from the
NOAA netCDF archive, with the GloTEC foF2 anomaly vs cached IRI, the quality flag, and the lag.

    uv run train/glotec_tokens.py 2025-06-15T12:00   # prints the token array shape for a smoke check

Token features (F_GLO = 5 + 1 + 1 + 6 + 1 = 14):
  lat/90, sin lon, cos lon, sin LT, cos LT | Δt/24 (lag, negative) | anomaly (foF2_glotec − IRI)/1.5 |
  quality flag one-hot (0..5) | log1p(TEC)/4
Cells: every cell with qf>0 at lag 0 (plus a 1-in-4 sample of qf=0 cells, since even those carried
information in Phase 3), at lags 0, −1, −6, −24 h (nearest 10-min step, only if that day's file exists).
Cap: MAX_GLO tokens, random subsample. Returns an empty (0, F_GLO) array when no GloTEC file exists.
"""
import datetime as dt
import os
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # train/train.py shadows the package name
from baselines.iri_cache import iri_points_cached  # noqa: E402
from build_samples import geo_feats  # noqa: E402

GLOTEC = Path(os.environ.get("GLOTEC_DIR", "/kass/forecast/glotec"))
LAGS_H = (0, 1, 6, 24)
# Measured 2026-09-05: the NOAA file is updated ~16 min after a step's nominal time; with our 15-min
# run cadence the newest usable step is ~25 min old. 30 min keeps training honest for the live case.
LATENCY_MIN = 30
F_GLO = 14
MAX_GLO = 1200
_cache = {}


def _load(day: dt.date):
    if day in _cache:
        return _cache[day]
    f = GLOTEC / f"GloTEC_TEC_{day:%Y_%m_%d}.nc"
    if not f.exists():
        return None  # not cached: the live service sees today's file appear later in the day
    with nc.Dataset(f) as ds:
        d = dict(lat=ds["latitude"][:].filled(np.nan), lon=ds["longitude"][:].filled(np.nan), t=np.asarray(ds["time"][:]).astype("int64"),
                 nm=np.asarray(ds["NmF2"][:]), tec=np.asarray(ds["TEC"][:]), qf=np.asarray(ds["quality_flag"][:]))
    if len(_cache) > 8:
        _cache.pop(next(iter(_cache)))
    _cache[day] = d
    return d


STALE_MAX_H = 3  # a lag's step may be up to this much older than nominal (late/missing NOAA file) before it is dropped


def _step_at_or_before(tl):
    """Newest GloTEC step with time <= tl within STALE_MAX_H, looking in tl's day file then the previous day's.
    Returns (day dict, index) or None."""
    day = tl.astype("datetime64[D]").astype(dt.date)
    for dday in (day, day - dt.timedelta(days=1)):
        d = _load(dday)
        if d is None:
            continue
        ok = np.nonzero(d["t"] <= tl.astype("int64"))[0]  # newest step at or before tl (= training's nearest-step choice on the 10-min grid)
        if len(ok) and tl.astype("int64") - d["t"][ok[-1]] <= STALE_MAX_H * 3600:
            return d, int(ok[-1])
    return None


def glotec_tokens(t, f107, rng, max_glo=MAX_GLO):
    """t: issue time (np.datetime64 or str); returns (N, F_GLO) float32."""
    t0 = np.datetime64(t, "s")
    out = []; keep0 = None
    for lag in LAGS_H:
        tl = t0 - np.timedelta64(lag * 3600 + LATENCY_MIN * 60, "s")  # lag, minus the assumed arrival latency
        step = _step_at_or_before(tl)
        if step is None:  # no file (e.g. NOAA's daily file not yet published) or nothing within STALE_MAX_H
            if lag == 0:
                return np.zeros((0, F_GLO), np.float32)  # lags are keyed to lag 0's cells; without it, run without GloTEC
            continue
        d, i = step
        qf, nm, tec = d["qf"][i], d["nm"][i], d["tec"][i]
        lon, lat = np.meshgrid(d["lon"], d["lat"])
        keep = qf > 0  # qf=0 cells are GloTEC's own extrapolation (no observations): excluded as a matter of principle (2026-09-09)
        keep &= np.isfinite(nm) & (nm > 0)
        if lag == 0:
            keep0 = keep
        else:
            keep = keep0 & np.isfinite(nm) & (nm > 0)  # same cells as lag 0 so the model can read trends
        la, lo = lat[keep], lon[keep]
        fof2 = np.sqrt(nm[keep] / 1.24e10)
        times = np.full(len(la), np.datetime64(int(d["t"][i]), "s"))
        iri = iri_points_cached(times, la, lo, f107)["fof2"]
        onehot = np.eye(6, dtype=np.float32)[np.clip(qf[keep], 0, 5)]
        dt_days = (np.datetime64(int(d["t"][i]), "s") - t0).astype(float) / 86400.0  # true age of the step used (= -(lag + latency) when the nominal step exists)
        tok = np.c_[geo_feats(la, lo, times), np.full(len(la), dt_days), (fof2 - iri) / 1.5, onehot, np.log1p(np.clip(tec[keep], 0, None)) / 4.0]
        out.append(tok.astype(np.float32))
    if not out:
        return np.zeros((0, F_GLO), np.float32)
    tok = np.concatenate(out)
    if len(tok) > max_glo:
        tok = tok[rng.choice(len(tok), max_glo, replace=False)]
    return tok


if __name__ == "__main__":
    import duckdb
    from data.drivers import attach_indices, f107_trailing
    from data.load import SNAPSHOT
    con = duckdb.connect(); attach_indices(con, SNAPSHOT)
    t = sys.argv[1] if len(sys.argv) > 1 else "2025-06-15T12:00"
    tok = glotec_tokens(np.datetime64(t), f107_trailing(con, t.replace("T", " ")), np.random.default_rng(0))
    print(tok.shape, "anomaly mean/std", tok[:, 6].mean().round(3), tok[:, 6].std().round(3), "qf dist", tok[:, 7:13].sum(0).astype(int), "lags", np.unique(np.round(tok[:, 5] * 24, 2)))
    assert tok.shape[1] == F_GLO and np.isfinite(tok).all()
