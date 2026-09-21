"""IRI-2020 via PyIRI. The climate baseline and the mean field every anomaly model sits on."""
import datetime as dt

import numpy as np
import PyIRI.sh_library as sh

import baselines.pyiri_patch  # noqa: F401  (memoises Apex transforms; ~10x faster)

# hmF2 model: production's map generator (irimap) uses Shubin-2015, its point driver AMTB-2013,
# PyIRI's legacy entry point BSE-1979. SHU2015 is PyIRI's default and is COSMIC-derived like our RO truth.
HMF2_MODEL = "SHU2015"


def iri_day(day: dt.date, hours, lat, lon, f107, ursi=True, hmf2_model=None):
    """foF2 [MHz], hmF2 [km], MUF(3000) [MHz] for one UTC day.

    hours: (T,) UT hours; lat, lon: (G,) degrees. Returns three (T, G) arrays.
    """
    F2 = sh.IRI_density_1day(
        day.year, day.month, day.day, np.asarray(hours, float), np.asarray(lon, float),
        np.asarray(lat, float), np.array([300.0]), float(f107),
        foF2_coeff="URSI" if ursi else "CCIR", hmF2_model=hmf2_model or HMF2_MODEL, old_output=True,
    )[0]
    return F2["fo"], F2["hm"], F2["fo"] * F2["M3000"]


def iri_points(times, lat, lon, f107, time_res_min=15, hmf2_model=None):
    """Per-point evaluation: times (N,) datetimes, lat/lon (N,). Returns dict of (N,) arrays.

    Times are rounded to `time_res_min` so the hours x points product stays small; IRI is smooth
    enough in time that 15 min costs nothing measurable.

    Groups by UTC day so PyIRI's per-day setup runs once per day, and evaluates every point of
    that day at its own hour (PyIRI computes the full hours x points product, so a day with many
    points is one call; we take the diagonal).
    """
    times = np.asarray(times, dtype="datetime64[s]")
    step = np.timedelta64(time_res_min * 60, "s")
    times = (times + step // 2) - ((times + step // 2) - np.datetime64(0, "s")) % step
    out = {k: np.full(len(times), np.nan) for k in ("fof2", "hmf2", "mufd")}
    days = times.astype("datetime64[D]")
    for day in np.unique(days):
        idx = np.nonzero(days == day)[0]
        hours = (times[idx] - day).astype("timedelta64[s]").astype(float) / 3600.0
        # Distinct hours to keep the T x G product small; map each point to its hour row.
        uh, inv = np.unique(hours, return_inverse=True)
        fo, hm, muf = iri_day(day.astype(dt.date), uh, lat[idx], lon[idx], f107 if np.isscalar(f107) else f107[idx].mean(), hmf2_model=hmf2_model)
        out["fof2"][idx] = fo[inv, np.arange(len(idx))]
        out["hmf2"][idx] = hm[inv, np.arange(len(idx))]
        out["mufd"][idx] = muf[inv, np.arange(len(idx))]
    return out


if __name__ == "__main__":
    t = np.array(["2024-03-15T12:00", "2024-03-15T00:00", "2024-03-16T12:00"], dtype="datetime64[s]")
    r = iri_points(t, np.array([41.8, 41.8, 41.8]), np.array([12.5, 12.5, 12.5]), 150.0)
    assert r["fof2"][0] > r["fof2"][1] > 3, r  # Rome: noon > midnight
    assert 200 < r["hmf2"][0] < 400 and r["mufd"][0] > r["fof2"][0]
    assert abs(r["fof2"][0] - r["fof2"][2]) < 0.5  # consecutive days similar
    print("ok", {k: np.round(v, 2) for k, v in r.items()})
