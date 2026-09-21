"""Per-day cache of global IRI maps, so PyIRI's expensive SH evaluation runs once per (day, driver, hmF2 model).

Grid: half-hourly, 2° lat x 2° lon (91 x 181), float32 fof2/hmf2/mufd, F10.7 quantised to F107_STEP SFU
with linear interpolation between the two neighbouring levels (IRI itself is linear in the solar
index between its min/max coefficient sets). Trilinear interpolation to arbitrary (time, lat, lon). IRI is smooth at these scales; interpolation error is
far below the 1.4 MHz climatology error we measure against.
"""
import datetime as dt
import os
from pathlib import Path

import numpy as np

from baselines.iri import HMF2_MODEL, iri_day

CACHE = Path(os.environ.get("IRI_CACHE", "/kass/forecast/iri_cache"))
LAT = np.arange(-90, 91, 2.0)
LON = np.arange(-180, 181, 2.0)
F107_STEP = 10
HOURS = np.append(np.arange(0, 24, 0.5), 23.999)  # PyIRI rejects hour 24; 23.999 closes the day
DT = 0.5
_mem = {}


def _key(day, f107_level, hmf2_model):
    return f"{hmf2_model}/{day.isoformat()}_{int(f107_level)}"


def _iri_maps_level(day: dt.date, f107_level, hmf2_model):
    """(49, 3, 91, 181) float32 [fof2, hmf2, mufd], hours 0..23.5 + 23.999, at one quantised F10.7. Cached in memory and on disk."""
    k = _key(day, f107_level, hmf2_model)
    if k in _mem:
        return _mem[k]
    f = CACHE / f"{k}.npz"
    if f.exists():
        m = np.load(f)["maps"]
    else:
        lon, lat = np.meshgrid(LON, LAT)
        fo, hm, muf = iri_day(day, HOURS, lat.ravel(), lon.ravel(), float(f107_level), hmf2_model=hmf2_model)
        m = np.stack([fo, hm, muf], 1).reshape(len(HOURS), 3, len(LAT), len(LON)).astype(np.float32)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f"{f.stem}.{os.getpid()}.tmp.npz")  # unique: parallel builders may race
        np.savez_compressed(tmp, maps=m)
        if f.exists():  # someone else finished first; theirs is identical
            tmp.unlink()
        else:
            tmp.rename(f)
    if len(_mem) > 64:
        _mem.pop(next(iter(_mem)))
    _mem[k] = m
    return m


def _foe_maps_level(day: dt.date, f107_level):
    """(49, 91, 181) float32 foE [MHz] on the same grid/hours as the IRI maps. Cached like them (~10 s to build)."""
    k = f"foe/{day.isoformat()}_{int(f107_level)}"
    if k in _mem:
        return _mem[k]
    f = CACHE / f"{k}.npz"
    if f.exists():
        m = np.load(f)["foe"]
    else:
        import PyIRI.sh_library as sh
        import baselines.pyiri_patch  # noqa: F401
        lon, lat = np.meshgrid(LON, LAT)
        E = sh.IRI_density_1day(day.year, day.month, day.day, HOURS, lon.ravel(), lat.ravel(), np.array([110.0]), float(f107_level), old_output=True)[2]
        m = E["fo"].reshape(len(HOURS), len(LAT), len(LON)).astype(np.float32)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f"{f.stem}.{os.getpid()}.tmp.npz"); np.savez_compressed(tmp, foe=m)
        if f.exists():
            tmp.unlink()
        else:
            tmp.rename(f)
    _mem[k] = m
    return m


def foe_points_cached(times, lat, lon, f107):
    """foE at arbitrary (time, lat, lon), trilinear from the cached 2° half-hourly maps; F10.7 blended like iri_maps."""
    times = np.asarray(times, dtype="datetime64[s]"); lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    out = np.full(len(times), np.nan, np.float32)
    lo_lvl = int(np.floor(f107 / F107_STEP)) * F107_STEP; w = (f107 - lo_lvl) / F107_STEP
    for day in np.unique(times.astype("datetime64[D]")):
        idx = np.nonzero(times.astype("datetime64[D]") == day)[0]
        hour = (times[idx] - day).astype("timedelta64[s]").astype(float) / 3600.0
        m = _foe_maps_level(day.astype(dt.date), lo_lvl)
        if w > 1e-6:
            m = (1 - w) * m + w * _foe_maps_level(day.astype(dt.date), lo_lvl + F107_STEP)
        out[idx] = _interp(m[:, None], hour, lat[idx], lon[idx])[:, 0]
    return out.astype(float)


def iri_maps(day: dt.date, f107, hmf2_model=None):
    """Maps at an arbitrary F10.7: linear blend of the two neighbouring quantised levels."""
    hmf2_model = hmf2_model or HMF2_MODEL
    lo = int(np.floor(f107 / F107_STEP)) * F107_STEP
    w = (f107 - lo) / F107_STEP
    a = _iri_maps_level(day, lo, hmf2_model)
    if w < 1e-6:
        return a
    return (1 - w) * a + w * _iri_maps_level(day, lo + F107_STEP, hmf2_model)


def _interp(m, hour, lat, lon):
    """Trilinear interpolation of (49,3,91,181) maps at (N,) hour/lat/lon -> (N,3)."""
    th = np.clip(hour, 0, 23.999); ti = np.minimum((th / DT).astype(int), len(HOURS) - 2); tf = (th - HOURS[ti]) / (HOURS[ti + 1] - HOURS[ti])
    la = (np.clip(lat, -90, 90) + 90) / 2; li = np.minimum(la.astype(int), len(LAT) - 2); lf = la - li
    lo = np.mod(lon + 180, 360) / 2; oi = np.minimum(lo.astype(int), len(LON) - 2); of = lo - oi
    out = np.zeros((len(hour), 3), np.float32)
    for dti, wt in ((0, 1 - tf), (1, tf)):
        for dla, wl in ((0, 1 - lf), (1, lf)):
            for dlo, wo in ((0, 1 - of), (1, of)):
                out += (wt * wl * wo)[:, None] * m[ti + dti, :, li + dla, oi + dlo]
    return out


def iri_points_cached(times, lat, lon, f107, hmf2_model=None):
    """Same contract as baselines.iri.iri_points, via the map cache. f107 scalar per call."""
    times = np.asarray(times, dtype="datetime64[s]")
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    out = np.full((len(times), 3), np.nan, np.float32)
    days = times.astype("datetime64[D]")
    for day in np.unique(days):
        idx = np.nonzero(days == day)[0]
        hour = (times[idx] - day).astype("timedelta64[s]").astype(float) / 3600.0
        out[idx] = _interp(iri_maps(day.astype(dt.date), f107, hmf2_model), hour, lat[idx], lon[idx])
    return {"fof2": out[:, 0].astype(float), "hmf2": out[:, 1].astype(float), "mufd": out[:, 2].astype(float)}


if __name__ == "__main__":
    import time
    from baselines.iri import iri_points
    rng = np.random.default_rng(0)
    n = 400
    t = np.datetime64("2024-03-15T00:00:00") + rng.integers(0, 86400, n).astype("timedelta64[s]")
    la = rng.uniform(-80, 80, n); lo = rng.uniform(-180, 180, n)
    s = time.time(); a = iri_points(t, la, lo, 153.0, time_res_min=1); ta = time.time() - s
    s = time.time(); iri_points_cached(t, la, lo, 153.0); tb = time.time() - s  # builds two levels
    s = time.time(); b = iri_points_cached(t, la, lo, 153.0); tc = time.time() - s
    for v in ("fof2", "hmf2", "mufd"):
        d = b[v] - a[v]
        print(f"{v}: interp-exact bias {d.mean():+.3f} rms {np.sqrt((d**2).mean()):.3f} max {np.abs(d).max():.3f}")
    print(f"exact {ta:.1f}s, cache build {tb:.1f}s, cached {tc*1000:.0f} ms")
    assert np.sqrt(((b["fof2"] - a["fof2"]) ** 2).mean()) < 0.03
