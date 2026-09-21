"""Build training samples for the anomaly model from the replay harness's point-in-time protocol.

    uv run train/build_samples.py --start 2024-01-01 --end 2024-02-01 --step-hours 3 --out /kass/forecast/samples/2024

One .npz per issue time T:
  tok   (N, F_TOK)  ionosonde observations in [T-24h, T-15min]; features below; anomalies vs IRI
  qry   (M, F_QRY)  query points = every observation in (T, T+24h] at all stations (+ RO), features below
  tgt   (M, 3)      anomaly targets (obs - IRI) for fof2, hmf2, mufd; NaN where absent
  glob  (F_GLOB,)   indices known at T
  meta: held (M,) bool = query's cluster was withheld from tok (holdout targets); qkind (M,) 0=iono 1=ro; qcluster (M,)
  identity: qtime (unix s), qlat, qlon, qid (station id or -1), iri_q (M,3) raw IRI at queries, issue (unix s)
Anomalies are normalised by ANOM_SCALE so all three variables are O(1).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.iri_cache import iri_points_cached  # noqa: E402
from data.drivers import attach_indices, f107_trailing  # noqa: E402
from data.load import SNAPSHOT, connect, inputs_at, targets_after  # noqa: E402

VARS = ("fof2", "hmf2", "mufd")
ANOM_SCALE = np.array([1.5, 40.0, 4.5])  # ~RMS of obs-IRI per variable
IRI_SCALE = np.array([10.0, 300.0, 30.0])
HOLDOUT_FRAC = 0.2
MAX_TOK = 2048
GLOTEC = [False]  # set by --glotec: add GloTEC cell tokens (tok_g) where the archive has the day
POOL = [False]  # set by --pool: pool_hourly() the ionosonde rows before tokenising
SPOT_RES = [60]  # set by --spot-res; recorded in each sample so train.py can put it in the checkpoint
SPOTS = [False]  # set by --spots: add WSPR/FT8 activity tokens (tok_s) from the hourly aggregates

# token features: lat/90, sin lon, cos lon, sin LT, cos LT, dt_h/24, iri fof2/hmf2/mufd (scaled), anomaly x3, present x3, cs/100
F_TOK = 1 + 2 + 2 + 1 + 3 + 3 + 3 + 1
# query features: lat/90, sin lon, cos lon, sin LT, cos LT, lead_h/24, iri x3 (scaled), kind
F_QRY = 1 + 2 + 2 + 1 + 3 + 1


def local_time(times, lon):
    h = (times.astype("datetime64[s]") - times.astype("datetime64[D]")).astype(float) / 3600.0
    return (h + lon / 15.0) % 24.0


def geo_feats(lat, lon, times):
    lt = local_time(times, lon) / 24.0 * 2 * np.pi
    lonr = np.radians(lon)
    return np.c_[lat / 90.0, np.sin(lonr), np.cos(lonr), np.sin(lt), np.cos(lt)]


QSTATE = [False]  # --qstate: append the nearest input station's latest state to every query (F_QSTATE columns)
F_QSTATE = 8
QSTATE_RADIUS_KM = 1500.0; QSTATE_L_KM = 500.0; QSTATE_MAX_AGE_H = 3.0  # Gaussian kernel, L 500 km (w=e^-9 at the 1500 km cut), so the feature is continuous in space


def query_state(s_lat, s_lon, s_time, s_anom, t0, q_lat, q_lon):
    """Kernel-weighted latest state of the nearby input stations for each query point -> (M, F_QSTATE) float32:
    [weight, dist_nearest/1500 km (1 = none in reach), age/24 h, latest anomaly fof2/hmf2/mufd (normalised), fof2 anomaly change over the
    last 1 h and 3 h]. Each station with data in the last QSTATE_MAX_AGE_H contributes with w = exp(-(d/L)^2),
    L = QSTATE_L_KM, zero beyond QSTATE_RADIUS_KM; the state columns are the w-weighted mean over stations and the
    first column is the summed weight (clipped to 1). Everything is continuous in space, so maps stay smooth (the
    first version used a hard 1000 km cut and nearest-station-wins, which drew discs around every station).
    Inputs are the (pooled) token rows the model sees, so training and the service agree by construction."""
    out = np.zeros((len(q_lat), F_QSTATE), np.float32); out[:, 1] = 1.0
    if len(s_lat) == 0:
        return out
    t0 = np.datetime64(t0, "s"); st = np.asarray(s_time, dtype="datetime64[s]")
    age = (t0 - st).astype(float) / 3600.0
    key = np.round(np.c_[s_lat, s_lon], 2)
    stations, inv = np.unique(key, axis=0, return_inverse=True)
    S = len(stations); latest = np.full(S, np.inf); a_now = np.zeros((S, 3)); a_1h = np.zeros(S); a_3h = np.zeros(S)
    for j in range(S):
        rows = np.nonzero(inv == j)[0]; rows = rows[np.argsort(age[rows])]
        r0 = rows[0]; latest[j] = age[r0]; a_now[j] = s_anom[r0]
        for back, dst in ((1.0, a_1h), (3.0, a_3h)):
            cand = rows[np.abs(age[rows] - (age[r0] + back)) <= 0.75]
            if len(cand):
                dst[j] = a_now[j, 0] - s_anom[cand[np.argmin(np.abs(age[cand] - (age[r0] + back)))], 0]
    ok = latest <= QSTATE_MAX_AGE_H
    if not ok.any():
        return out
    sl, so = np.radians(stations[ok, 0]), np.radians(stations[ok, 1]); ql, qo = np.radians(np.asarray(q_lat, float))[:, None], np.radians(np.asarray(q_lon, float))[:, None]
    d = 6371 * np.arccos(np.clip(np.sin(ql) * np.sin(sl) + np.cos(ql) * np.cos(sl) * np.cos(qo - so), -1, 1))  # (M, S_ok)
    w = np.exp(-(d / QSTATE_L_KM) ** 2); w[d > QSTATE_RADIUS_KM] = 0.0
    wsum = w.sum(1); has = wsum > 1e-6
    if not has.any():
        return out
    idx = np.nonzero(ok)[0]; c = np.minimum(wsum[has], 1.0)  # weighted means are scaled by the clipped weight sum so every
    wn = w[has] / wsum[has][:, None] * c[:, None]              # column tapers to 0 with distance instead of stepping at the cut
    out[:, 1] = 1.0; out[has, 1] = np.minimum(d[has].min(1) / QSTATE_RADIUS_KM, 1.0)  # 1 = no station within reach
    out[has, 0] = c
    out[has, 2] = wn @ latest[idx] / 24.0
    out[has, 3:6] = wn @ a_now[idx]
    out[has, 6] = wn @ a_1h[idx]; out[has, 7] = wn @ a_3h[idx]
    return out


def query_feats(times, lat, lon, iri_vals, kind):
    """Query features exactly as training used them. iri_vals (M,3) raw; kind (M,) 0 iono / 1 ro / 2 grid."""
    times = np.asarray(times, dtype="datetime64[s]")
    return np.c_[geo_feats(np.asarray(lat, float), np.asarray(lon, float), times), np.zeros(len(times)), iri_vals / IRI_SCALE,
                 (np.asarray(kind) == 1).astype(float)].astype(np.float32)


def pool_hourly(inp, t0):
    """One row per (station, hour of age before t0): mean time / values / cs over the rows in that hour, plus n.
    Most stations report every 5-15 min, so this cuts ~6-10k rows/day to ~1k tokens without dropping any station-hour;
    the token cap then stops discarding the freshest rows at busy stations (the suspected 0-4 h gap to the GP)."""
    inp = inp.copy(); inp["cs"] = np.where(inp.cs < 0, 0.85, inp.cs / 100.0)  # normalise first so the mean is meaningful
    age_h = (t0 - inp["time"].to_numpy().astype("datetime64[s]")).astype(float) / 3600.0
    inp["_b"] = np.floor(age_h).astype(int); inp["_t"] = inp["time"].to_numpy().astype("datetime64[s]").astype("int64")
    g = inp.groupby(["station_id", "_b"], sort=False)
    out = g.agg(cluster=("cluster", "first"), lat=("lat", "first"), lon=("lon", "first"), _t=("_t", "mean"), cs=("cs", "mean"),
                **{v: (v, "mean") for v in VARS}, n=("_t", "size")).reset_index()
    out["time"] = out.pop("_t").round().astype("int64").astype("datetime64[s]"); out["cs"] = out["cs"] * 100.0  # back to the raw convention
    return out.drop(columns="_b")


def tokens_at(con, t, f107, rng, holdout_frac=HOLDOUT_FRAC, max_tok=MAX_TOK, inp=None, pool=False):
    """Ionosonde tokens for issue time t (inputs in [t-24h, t-15min]); returns (tok, held_clusters, inputs df).
    inp: optional prebuilt inputs frame (live mode) with station_id, cluster, time, lat, lon, cs, fof2, hmf2, mufd.
    pool: pool_hourly() before tokenising (must match how the checkpoint's samples were built: ckpt args['pool'])."""
    t0 = np.datetime64(t, "s")
    if inp is None:
        inp = inputs_at(con, t, hours=24).df()
    clusters = np.unique(inp["cluster"])
    held = set(rng.choice(clusters, int(round(holdout_frac * len(clusters))), replace=False)) if holdout_frac > 0 else set()
    inp = inp[~inp["cluster"].isin(held)]
    if pool:
        inp = pool_hourly(inp, t0)
    if "n" not in inp:
        inp = inp.assign(n=1)
    if len(inp) > max_tok:
        inp = inp.sample(max_tok, random_state=int(rng.integers(2**31)))
    it = inp["time"].to_numpy().astype("datetime64[s]")
    iri_i = iri_points_cached(it, inp.lat.to_numpy(), inp.lon.to_numpy(), f107)
    obs = np.c_[[inp[v].to_numpy(dtype=float) for v in VARS]].T
    iri_v = np.c_[iri_i["fof2"], iri_i["hmf2"], iri_i["mufd"]]
    present = np.isfinite(obs)
    anom = np.where(present, (obs - iri_v) / ANOM_SCALE, 0.0)
    dt_h = (it - t0).astype(float) / 3600.0
    cs = np.where(inp.cs.to_numpy() < 0, 0.85, inp.cs.to_numpy() / 100.0)
    tok = np.c_[geo_feats(inp.lat.to_numpy(), inp.lon.to_numpy(), it), dt_h / 24.0, iri_v / IRI_SCALE, anom, present.astype(float), cs,
                np.log1p(inp.n.to_numpy(float)) / 3.0].astype(np.float32)  # 17th column: rows pooled into this token (0.23 when unpooled)
    inp = inp.assign(iri_fof2=iri_v[:, 0], anom_fof2=obs[:, 0] - iri_v[:, 0], anom_n=list(anom))  # anom_n: normalised (fof2,hmf2,mufd) for query_state
    return tok, held, inp


def build(con, t, rng, f107, held_stations=None):
    t0 = np.datetime64(t, "s")
    inp = inputs_at(con, t, hours=24).df()
    tg = targets_after(con, t).df()
    if len(inp) < 20 or len(tg) < 20:
        return None
    clusters = np.unique(inp["cluster"])
    if held_stations is not None:  # reproduce an external holdout (production's) by cluster, as replay.py does
        held = set(inp.loc[inp["station_id"].isin(held_stations), "cluster"])
    else:
        held = set(rng.choice(clusters, int(round(HOLDOUT_FRAC * len(clusters))), replace=False))
    inp = inp[~inp["cluster"].isin(held)]
    if POOL[0]:
        inp = pool_hourly(inp, t0)
    if "n" not in inp:
        inp = inp.assign(n=1)
    if len(inp) > MAX_TOK:  # build-time cap; train.py --max-tok subsamples further at load time (token-budget study)
        inp = inp.sample(MAX_TOK, random_state=int(rng.integers(2**31)))
    # tokens
    it = inp["time"].to_numpy().astype("datetime64[s]")
    iri_i = iri_points_cached(it, inp.lat.to_numpy(), inp.lon.to_numpy(), f107)
    obs = np.c_[[inp[v].to_numpy(dtype=float) for v in VARS]].T  # (N,3)
    iri_v = np.c_[iri_i["fof2"], iri_i["hmf2"], iri_i["mufd"]]
    present = np.isfinite(obs)
    anom = np.where(present, (obs - iri_v) / ANOM_SCALE, 0.0)
    dt_h = (it - t0).astype(float) / 3600.0  # negative
    cs = np.where(inp.cs.to_numpy() < 0, 0.85, inp.cs.to_numpy() / 100.0)
    tok = np.c_[geo_feats(inp.lat.to_numpy(), inp.lon.to_numpy(), it), dt_h / 24.0, iri_v / IRI_SCALE, anom, present.astype(float), cs,
                np.log1p(inp.n.to_numpy(float)) / 3.0].astype(np.float32)
    # queries + targets
    qt = tg["time"].to_numpy().astype("datetime64[s]")
    iri_q = iri_points_cached(qt, tg.lat.to_numpy(), tg.lon.to_numpy(), f107)
    iri_qv = np.c_[iri_q["fof2"], iri_q["hmf2"], iri_q["mufd"]]
    lead = (qt - t0).astype(float) / 3600.0
    kind = (tg["kind"].to_numpy() == "ro").astype(float)
    qry = np.c_[geo_feats(tg.lat.to_numpy(), tg.lon.to_numpy(), qt), lead / 24.0, iri_qv / IRI_SCALE, kind].astype(np.float32)
    if QSTATE[0]:
        qry = np.c_[qry, query_state(inp.lat.to_numpy(), inp.lon.to_numpy(), it, anom, t0, tg.lat.to_numpy(), tg.lon.to_numpy())].astype(np.float32)
    tobs = np.c_[[tg[v].to_numpy(dtype=float) for v in VARS]].T
    tgt = ((tobs - iri_qv) / ANOM_SCALE).astype(np.float32)  # NaN where absent
    qcluster = tg["cluster"].to_numpy(dtype=float)
    held_q = np.isin(qcluster, list(held)) | (kind == 1)
    return dict(tok=tok, qry=qry, tgt=tgt, held=held_q, qkind=kind.astype(np.int8), qcluster=np.nan_to_num(qcluster, nan=-1).astype(np.int32),
                tok_g=(glotec_tokens(t0, f107, rng) if GLOTEC[0] else np.zeros((0, 14), np.float32)),
                tok_s=(spot_tokens(t0, rng) if SPOTS[0] else np.zeros((0, 29), np.float32)), pool=np.bool_(POOL[0]), spot_res=np.int16(SPOT_RES[0]), qstate=np.bool_(QSTATE[0]),
                # identity for the harness schema (eval/report.py pairs on issue_time, target_id, time, lat, lon)
                qtime=qt.astype("datetime64[s]").astype(np.int64), qlat=tg.lat.to_numpy(np.float32), qlon=tg.lon.to_numpy(np.float32),
                qid=np.nan_to_num(tg["target_id"].to_numpy(dtype=float), nan=-1).astype(np.int32),
                iri_q=iri_qv.astype(np.float32), issue=np.int64(t0.astype("datetime64[s]").astype(np.int64)))


def main():
    global MAX_TOK
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--step-hours", type=float, default=3); ap.add_argument("--out", required=True)
    ap.add_argument("--holdouts", help="parquet with issue_time, station_id: build at these issue times with these stations withheld")
    ap.add_argument("--glotec", action="store_true", help="add GloTEC tokens (tok_g) from /kass/forecast/glotec where available")
    ap.add_argument("--spots", action="store_true", help="add WSPR/FT8 spot-activity tokens (tok_s) from the hourly aggregates")
    ap.add_argument("--spots-cp", action="store_true", help="with --spots: add control-point slots for >3000 km paths (52-feature tokens)")
    ap.add_argument("--pool", action="store_true", help="one token per (station, hour) with mean values and a count feature (pool_hourly)")
    ap.add_argument("--spot-res", type=int, default=60, help="spot aggregate resolution in minutes: 60 = clock hours (v5..v7), 5 = bins ending at T-15 min")
    ap.add_argument("--max-spot", type=int, help="spot token cap (default 3000; control-point builds need ~6000 so midpoint tokens are not displaced)")
    ap.add_argument("--qstate", action="store_true", help="append the nearest input station's latest state to every query (query_state; F_QRY 10 -> 18)")
    ap.add_argument("--max-tok", type=int, default=MAX_TOK, help="ionosonde token cap per sample (default 2048; 8192 keeps ~everything)")
    a = ap.parse_args()
    GLOTEC[0] = a.glotec; SPOTS[0] = a.spots; MAX_TOK = a.max_tok; POOL[0] = a.pool; QSTATE[0] = a.qstate
    if a.spots:
        global spot_tokens
        sys.path.insert(0, str(Path(__file__).resolve().parent)); from spot_tokens import spot_tokens, CP  # noqa: E402
        CP[0] = a.spots_cp; sys.modules["spot_tokens"].RES[0] = a.spot_res; SPOT_RES[0] = a.spot_res
        if a.max_spot:
            sys.modules["spot_tokens"].MAX_SPOT = a.max_spot
    if a.glotec:
        global glotec_tokens
        sys.path.insert(0, str(Path(__file__).resolve().parent)); from glotec_tokens import glotec_tokens  # noqa: E402
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    con = connect(); attach_indices(con, SNAPSHOT)
    daily = con.execute("SELECT date, Ap, f107_obs FROM daily").df().set_index("date")
    ap8 = con.execute("SELECT date, ap3h FROM daily ORDER BY date").df()
    ap_series = pd.Series(np.concatenate(ap8.ap3h.to_list()).astype(float),
                          index=np.repeat(pd.to_datetime(ap8.date), 8) + pd.to_timedelta(np.tile(np.arange(8) * 3, len(ap8)), unit="h"))
    held_by_t = None
    if a.holdouts:
        h = pd.read_parquet(a.holdouts); h = h[(h.issue_time >= a.start) & (h.issue_time < a.end)]
        held_by_t = h.groupby("issue_time")["station_id"].apply(set).to_dict()
        issue_times = sorted(held_by_t)
    else:
        issue_times = pd.date_range(a.start, a.end, freq=f"{int(a.step_hours * 60)}min", inclusive="left")
    n = 0
    for t in issue_times:
        f = out / f"{t.strftime('%Y%m%dT%H%M')}.npz"
        if f.exists():
            continue
        f107 = f107_trailing(con, str(t))
        rng = np.random.default_rng(int(t.value // 10**9) % 2**32)
        s = build(con, str(t), rng, f107, held_stations=None if held_by_t is None else held_by_t[t])
        if s is None:
            continue
        slot = t.floor("3h")
        ap_now = float(ap_series.get(slot, np.nan)); ap_max = float(ap_series.loc[slot - pd.Timedelta(hours=21):slot].max())
        d = t.date()
        f107d = float(daily.f107_obs.get(pd.Timestamp(d) - pd.Timedelta(days=1), np.nan))
        if not np.isfinite(f107d):  # GFZ daily gaps: fall back to the 81-day mean
            f107d = f107
        doy = t.dayofyear / 365.25 * 2 * np.pi
        s["glob"] = np.array([f107 / 200.0, f107d / 200.0, np.log1p(np.nan_to_num(ap_now, nan=ap_max)) / 5, np.log1p(ap_max) / 5, np.sin(doy), np.cos(doy)], np.float32)
        tmp = f.with_suffix(f".{os.getpid()}.tmp.npz"); np.savez_compressed(tmp, **s); os.replace(tmp, f)  # atomic: a second builder over the same range can't corrupt it
        n += 1
        if n % 50 == 0:
            print(f"{t}  {n} written", flush=True)
    print(f"done: {n} samples -> {out}")


if __name__ == "__main__":
    main()
