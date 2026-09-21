"""Model protocol and the trivial baselines (PLAN.md Phase 2).

A model is `forecast(ctx) -> dict of (N,) arrays for fof2, hmf2, mufd` at ctx.targets. NaN = no forecast.
ctx has: con, t (issue time, str), inputs (DataFrame: station_id, cluster, time, lat, lon, fof2, hmf2, mufd),
targets (DataFrame: kind, target_id, cluster, time, lat, lon), iri (dict of arrays at targets), f107.
"""
import numpy as np

VARS = ("fof2", "hmf2", "mufd")


def iri(ctx):
    return {v: ctx.iri[v] for v in VARS}


def persistence(ctx):
    """Last input value per station, held flat. Undefined off-station (RO, held-out clusters)."""
    last = ctx.inputs.sort_values("time").groupby("station_id").last()
    out = {v: np.full(len(ctx.targets), np.nan) for v in VARS}
    idx = ctx.targets["target_id"].map(lambda s: last.index.get_loc(s) if s in last.index else -1).to_numpy()
    ok = idx >= 0
    for v in VARS:
        out[v][ok] = last[v].to_numpy()[idx[ok]]
    return out


def iri_essn(ctx):
    """IRI driven by production's eSSN fit at T (what `irimap` does). Falls back to IRI if no fit."""
    from baselines.iri_cache import iri_points_cached as iri_points
    from data.drivers import essn_sfi

    sfi = essn_sfi(ctx.con, ctx.t)
    if sfi is None:
        return {v: np.full(len(ctx.targets), np.nan) for v in VARS}
    return iri_points(ctx.targets["time"].to_numpy(), ctx.targets["lat"].to_numpy(), ctx.targets["lon"].to_numpy(), sfi)


def _gc_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    c = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2)
    return 6371.0 * np.arccos(np.clip(c, -1, 1))


def make_anomaly_decay(tau_h=48.0, scale_km=2000.0, window_h=2.0, shrink=1.0, base="iri"):
    """IRI + station anomalies at T, spread by a Gaussian distance kernel, decayed with lead time.

    shrink: added to the kernel weight sum, so far from every station the anomaly goes to zero
    (GP-style shrinkage) instead of extrapolating the nearest station's anomaly at full size.
    base: "iri" (F10.7-driven) or "essn" (production eSSN-driven), i.e. what the anomaly is measured against.
    """
    from baselines.iri_cache import iri_points_cached as iri_points
    from data.drivers import essn_sfi

    def anomaly_decay(ctx):
        t0 = np.datetime64(ctx.t, "s")
        recent = ctx.inputs[ctx.inputs["time"].to_numpy() >= t0 - np.timedelta64(int(window_h * 3600), "s")]
        f107 = ctx.f107
        if base == "essn":
            f107 = essn_sfi(ctx.con, ctx.t)
            if f107 is None:
                return {v: np.full(len(ctx.targets), np.nan) for v in VARS}
            out = iri_points(ctx.targets["time"].to_numpy(), ctx.targets["lat"].to_numpy(), ctx.targets["lon"].to_numpy(), f107)
        else:
            out = {v: ctx.iri[v].copy() for v in VARS}
        if len(recent) == 0:
            return out
        base_v = iri_points(recent["time"].to_numpy(), recent["lat"].to_numpy(), recent["lon"].to_numpy(), f107)
        anom = recent[["cluster", "lat", "lon"]].copy()
        for v in VARS:
            anom[v] = recent[v].to_numpy() - base_v[v]
        anom = anom.groupby("cluster").mean()  # one anomaly per site
        w = np.exp(-0.5 * (_gc_km(ctx.targets["lat"].to_numpy()[:, None], ctx.targets["lon"].to_numpy()[:, None],
                                  anom["lat"].to_numpy()[None, :], anom["lon"].to_numpy()[None, :]) / scale_km) ** 2)
        lead_h = (ctx.targets["time"].to_numpy().astype("datetime64[s]") - t0).astype(float) / 3600.0
        decay = np.exp(-lead_h / tau_h)
        for v in VARS:
            a = anom[v].to_numpy()
            ok = np.isfinite(a)
            if not ok.any():
                continue
            wk = w[:, ok]
            out[v] = out[v] + decay * (wk @ a[ok]) / (wk.sum(1) + shrink + 1e-3)  # +eps: no stations -> IRI
        return out

    return anomaly_decay


MODELS = {"iri": iri, "iri_essn": iri_essn, "persistence": persistence, "anomaly_decay": make_anomaly_decay(),
          "essn_anomaly": make_anomaly_decay(base="essn")}
# Tuning grid for Phase 2 (PLAN.md): run with --models iri,$(grid names), pick on val holdout foF2.
for _tau in (6, 12, 24, 48):
    for _L in (500, 1000, 2000):
        for _sh in (0.0, 0.3, 1.0):
            MODELS[f"ad_t{_tau}_L{_L}_s{_sh}"] = make_anomaly_decay(_tau, _L, shrink=_sh)
# Second pass: first grid's optimum (t48, L2000, s1.0) was on the edge in every dimension.
for _tau in (48, 96, 1e9):
    for _L in (2000, 3000, 5000):
        for _sh in (1.0, 3.0):
            MODELS[f"ad2_t{int(min(_tau, 999))}_L{_L}_s{_sh}"] = make_anomaly_decay(_tau, _L, shrink=_sh)
