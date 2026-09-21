"""Map coherence gate (PLAN.md Phase 1): metrics on the anomaly field (forecast - IRI) of a rendered map set.

    uv run eval/coherence.py /kass/forecast/maps/2025-06-15_v0 /kass/forecast/maps/2025-06-15_anomaly_decay

Per map set (grids saved by forecast.py):
  spatial  - fraction of anomaly variance at zonal wavenumbers > k_station (scales finer than ~2000 km at
             mid-latitudes, i.e. finer than the station spacing); RMS gradient (MHz/100 km) far (>1500 km)
             from every input station vs near.
  temporal - RMS frame-to-frame change of the anomaly per hour, in the geographic frame and in the
             fixed-local-time frame (shift by 15°/h); fraction of pixels whose anomaly changes sign between
             consecutive frames.
Report side by side; the bounds are calibrated against the kernel / GP maps (PLAN.md).
"""
import sys
from pathlib import Path

import numpy as np

LAT = np.arange(-90, 91, 1.0); LON = np.arange(-180, 181, 1.0)
K_STATION = 20  # zonal wavenumber above which structure is finer than ~2000 km at 45° latitude


def gc_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    return 6371 * np.arccos(np.clip(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2), -1, 1))


def metrics(d, var=0):
    files = sorted(Path(d).glob("grid_lead*.npz"))
    zs = [np.load(f) for f in files]
    A = np.stack([(z["fc"][:, var] - z["iri"][:, var]).reshape(181, 361) for z in zs])  # (T, lat, lon)
    leads = np.array([float(z["lead_h"]) for z in zs]); st = zs[0]["stations"]
    # spatial: zonal power spectrum on mid-latitude rows, drop the duplicated lon 180
    a = A[:, 30:151, :360]; a = a - a.mean(2, keepdims=True)
    P = np.abs(np.fft.rfft(a, axis=2)) ** 2; ks = np.arange(P.shape[2])
    fine = P[:, :, ks > K_STATION].sum() / P[:, :, ks > 0].sum()
    lon, lat = np.meshgrid(LON, LAT)
    dmin = np.min(np.stack([gc_km(lat, lon, s[0], ((s[1] + 180) % 360) - 180) for s in st]), 0)
    gy, gx = np.gradient(A, axis=(1, 2))  # per degree
    grad = np.sqrt(gy ** 2 + (gx / np.maximum(np.cos(np.radians(lat)), 0.2)) ** 2) / 111.0 * 100  # per 100 km
    far, near = dmin > 1500, dmin < 500
    # temporal
    dt = np.diff(leads); geo = np.sqrt(((A[1:] - A[:-1]) ** 2).mean((1, 2))) / dt
    lt = []
    for i in range(1, len(A)):
        shift = int(round(15 * dt[i - 1]))  # sun moves west 15°/h: same local time = lon - 15*dt
        lt.append(np.sqrt(((np.roll(A[i], -shift, axis=1)[:, :360] - A[i - 1][:, :360]) ** 2).mean()) / dt[i - 1])
    big = np.abs(A) > 0.3  # sign flips only count where the anomaly is not negligible
    flips = np.mean([((np.sign(A[i]) != np.sign(A[i - 1])) & big[i] & big[i - 1]).sum() / max((big[i] & big[i - 1]).sum(), 1) for i in range(1, len(A))])
    return dict(anom_rms=float(np.sqrt((A ** 2).mean())), fine_scale_frac=float(fine), grad_far=float(grad[:, far].mean()) if far.any() else np.nan,
                grad_near=float(grad[:, near].mean()) if near.any() else np.nan, anom_far=float(np.abs(A[:, far]).mean()) if far.any() else np.nan,
                dtdt_geo=float(np.mean(geo)), dtdt_localtime=float(np.mean(lt)), sign_flip_frac=float(flips), n_leads=len(A))


def main():
    rows = {Path(d).name: metrics(d) for d in sys.argv[1:]}
    keys = list(next(iter(rows.values())))
    print(f"{'metric':>18} " + " ".join(f"{n:>26}" for n in rows))
    for k in keys:
        print(f"{k:>18} " + " ".join(f"{rows[n][k]:>26.4f}" for n in rows))
    print("\nanom_rms MHz | fine_scale_frac: anomaly variance at zonal k>20 (finer than station spacing) | grad_*: MHz per 100 km, far >1500 km / near <500 km from any station"
          "\nanom_far: mean |anomaly| >1500 km from stations | dtdt_*: RMS change per hour, geographic vs fixed-local-time frame | sign_flip_frac: pixels with |anomaly|>0.3 in both frames that change sign per step")


if __name__ == "__main__":
    main()
