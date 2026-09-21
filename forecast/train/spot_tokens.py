"""Spot-activity tokens (PLAN.md Phase 4b): one token per (5° midpoint cell, hour, source) from the hourly
WSPR / FT8 aggregates, carrying per-band activity and SNR anomalies against a climatological baseline.

    uv run train/spot_tokens.py baseline /kass/forecast/eval/spot_baseline.parquet 2019 2023   # build the baseline
    uv run train/spot_tokens.py 2024-06-15T12:00                                               # smoke: token shape

Token features (F_SPOT = 5 + 1 + 10 + 10 + 1 + 2 = 29):
  lat/90, sin lon, cos lon, sin LT, cos LT | Δt/24 (hour end, negative) | activity anomaly × 10 bands |
  SNR anomaly × 10 bands (0 where the band is absent) | log1p(total spots)/8 | source one-hot (WSPR, FT8)
Activity anomaly = log1p(n) − baseline median for (cell, band, UT hour, calendar month); baseline built from the
training years only, so evaluation years never see their own statistics.
Hours used: hours ending in [T−23h, T−1h] (the hour ending at T is not complete at T; 1 h assumed latency). Cap: MAX_SPOT tokens,
keeping the most recent hours first.
"""
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_samples import geo_feats  # noqa: E402

AGG = Path(os.environ.get("SPOT_AGG_DIR", "/kass/forecast/eval"))
BASELINE = Path(os.environ.get("SPOT_BASELINE", "/kass/forecast/eval/spot_baseline.parquet"))
BANDS = (1, 3, 5, 7, 10, 14, 18, 21, 24, 28)
SOURCES = {"wspr": 0, "psk": 1}
F_SPOT = 29
RES = [int(os.environ.get("SPOT_RES", 60))]  # 60: clock-hour aggregates ({src}_hourly_*); 5: 5-min aggregates ({src}_5min_*),
# hour bins ending at floor5(T-15 min), T-75.., so the newest spots are used instead of stopping at the last clock hour
LAG_MIN = 15
CP = [False]  # build_samples --spots-cp: add control-point slots (>3000 km paths at 1500 km from each end) -> 52 features
F_SPOT_CP = 52
MAX_SPOT = 3000
_con = None


def con():
    global _con
    if _con is None:
        _con = duckdb.connect(); _con.execute("SET memory_limit='8GB'; SET threads=2; SET TimeZone='UTC'")
        for src in list(SOURCES) + [f"{s}_cp" for s in SOURCES]:
            files = sorted(AGG.glob(f"{src}_{'hourly' if RES[0] == 60 else str(RES[0]) + 'min'}_*.parquet"))
            if files:
                _con.execute(f"CREATE VIEW {src} AS SELECT * FROM read_parquet({[str(f) for f in files]}, union_by_name=true)")
        if BASELINE.exists():
            _con.execute(f"CREATE VIEW base AS SELECT * FROM '{BASELINE}'")
    return _con


def build_baseline(out, y0, y1):
    """Baseline from years y0..y1 for WSPR (training years only, so eval years never see their own statistics).
    FT8 only exists from 2024-12, so its baseline uses all its years regardless (a mild, accepted leak)."""
    c = duckdb.connect(); c.execute("SET memory_limit='24GB'; SET TimeZone='UTC'")
    parts = []
    for src in list(SOURCES) + [f"{s}_cp" for s in SOURCES]:
        tag = "hourly" if RES[0] == 60 else f"{RES[0]}min"
        files = [str(f) for f in sorted(AGG.glob(f"{src}_{tag}_*.parquet")) if src.startswith("psk") or y0 <= int(f.stem.split('_')[-1]) <= y1]
        if not files:
            continue
        if RES[0] == 60:
            hourly = f"SELECT cell_lat, cell_lon, band, hour, n, snr_med AS snr FROM read_parquet({files}, union_by_name=true)"
        else:
            hourly = f"""SELECT cell_lat, cell_lon, band, date_trunc('hour', hour) AS hour, sum(n) AS n, sum(snr_sum) / sum(n) AS snr
                         FROM read_parquet({files}, union_by_name=true) GROUP BY 1, 2, 3, 4"""
        parts.append(f"""SELECT source, cell_lat, cell_lon, band, ut, month, median(logn) AS med_logn, median(snr) AS med_snr, count(*) AS n_hours
            FROM (SELECT '{src}' AS source, cell_lat, cell_lon, band, extract(hour FROM a.hour)::INT AS ut, extract(month FROM a.hour)::INT AS month, log(1 + n) AS logn, snr
                  FROM ({hourly}) a)
            GROUP BY source, cell_lat, cell_lon, band, ut, month""")
    c.execute(f"COPY ({' UNION ALL '.join(parts)}) TO '{out}' (FORMAT parquet)")
    print(c.execute(f"SELECT source, count(*) FROM '{out}' GROUP BY 1").fetchall())


def spot_tokens(t, rng, max_spot=None):
    """t: issue time (np.datetime64 or str) -> (N, F_SPOT) float32; empty when no aggregates/baseline."""
    if max_spot is None:
        max_spot = MAX_SPOT  # module global so build_samples --max-spot can override it
    t0 = pd.Timestamp(np.datetime64(t, "s"))
    c = con()
    if "base" not in {r[0] for r in c.execute("SELECT view_name FROM duckdb_views()").fetchall()}:
        return np.zeros((0, F_SPOT), np.float32)
    views = {r[0] for r in c.execute("SELECT view_name FROM duckdb_views()").fetchall()}
    nslot = 2 if CP[0] else 1  # slot 0 = path midpoints (1000-3000 km), slot 1 = control points of >3000 km paths
    frames = []
    for src, sid in SOURCES.items():
        for slot, view in enumerate([src, f"{src}_cp"][:nslot]):
            if view not in views:
                continue
            if RES[0] == 60:
                df = c.execute(f"""
                    SELECT a.hour, a.cell_lat, a.cell_lon, a.band, log(1 + a.n) - b.med_logn AS act, coalesce(a.snr_med - b.med_snr, 0) AS snr, a.n
                    FROM {view} a JOIN base b ON b.source = '{view}' AND b.cell_lat = a.cell_lat AND b.cell_lon = a.cell_lon AND b.band = a.band
                         AND b.ut = extract(hour FROM a.hour)::INT AND b.month = extract(month FROM a.hour)::INT
                    WHERE a.hour >= ? AND a.hour <= ? AND b.n_hours >= 6""", [t0 - pd.Timedelta(hours=24), t0 - pd.Timedelta(hours=2)]).df()
            else:
                # bins k = 0..23 end at E0 - k h, E0 = floor_res(T - LAG_MIN); `hour` below is the bin *start* so the
                # downstream code (hour_end = hour + 1 h) is shared; baseline keyed by the UT hour of the bin midpoint
                e0 = (t0 - pd.Timedelta(minutes=LAG_MIN)).floor(f"{RES[0]}min")
                df = c.execute(f"""
                    WITH bins AS (
                      SELECT cell_lat, cell_lon, band, floor(epoch(? - hour) / 3600 - 1e-9)::INT AS k, sum(n) AS n, sum(snr_sum) / sum(n) AS snr
                      FROM {view} WHERE hour >= ? AND hour < ? GROUP BY 1, 2, 3, 4)
                    SELECT ? - INTERVAL (a.k + 1) HOUR AS hour, a.cell_lat, a.cell_lon, a.band, log(1 + a.n) - b.med_logn AS act, coalesce(a.snr - b.med_snr, 0) AS snr, a.n
                    FROM bins a JOIN base b ON b.source = '{view}' AND b.cell_lat = a.cell_lat AND b.cell_lon = a.cell_lon AND b.band = a.band
                         AND b.ut = extract(hour FROM ? - INTERVAL (a.k) HOUR - INTERVAL 30 MINUTE)::INT
                         AND b.month = extract(month FROM ? - INTERVAL (a.k) HOUR - INTERVAL 30 MINUTE)::INT
                    WHERE b.n_hours >= 6""", [e0, e0 - pd.Timedelta(hours=24), e0, e0, e0, e0]).df()
            if df.empty:
                continue
            df["source"] = sid; df["slot"] = slot; frames.append(df)
    if not frames:
        return np.zeros((0, F_SPOT_CP if CP[0] else F_SPOT), np.float32)
    df = pd.concat(frames)
    # one token per (source, cell, hour); per slot: 10 band activity anomalies, 10 SNR anomalies, log total, presence
    bi = {b: i for i, b in enumerate(BANDS)}
    keys = df.groupby(["source", "cell_lat", "cell_lon", "hour"]).size().reset_index()[["source", "cell_lat", "cell_lon", "hour"]]
    act = np.zeros((nslot, len(keys), 10), np.float32); snr = np.zeros((nslot, len(keys), 10), np.float32); total = np.zeros((nslot, len(keys)), np.float64)
    kidx = {k: i for i, k in enumerate(zip(keys.source, keys.cell_lat, keys.cell_lon, keys.hour))}
    for r in df.itertuples(index=False):
        i = kidx[(r.source, r.cell_lat, r.cell_lon, r.hour)]; j = bi.get(int(r.band))
        if j is not None:
            act[r.slot, i, j] = r.act; snr[r.slot, i, j] = r.snr; total[r.slot, i] += r.n
    lat = keys.cell_lat.to_numpy(float) + 2.5; lon = keys.cell_lon.to_numpy(float) + 2.5
    hour_end = (keys.hour + pd.Timedelta(hours=1)).to_numpy().astype("datetime64[s]")
    dt_h = (hour_end - np.datetime64(t0, "s")).astype(float) / 3600.0
    cols = [geo_feats(lat, lon, hour_end), dt_h / 24.0, act[0], snr[0], np.log1p(total[0]) / 8.0]
    if CP[0]:
        cols += [act[1], snr[1], np.log1p(total[1]) / 8.0, (total[0] > 0).astype(np.float32), (total[1] > 0).astype(np.float32)]
    tok = np.c_[tuple(cols) + (np.eye(2, dtype=np.float32)[keys.source.to_numpy(int)],)].astype(np.float32)
    if len(tok) > max_spot:  # keep the most recent hours preferentially: sort by -dt and take the head
        order = np.argsort(-dt_h); tok = tok[order[:max_spot]]
    return tok


if __name__ == "__main__":
    if sys.argv[1] == "baseline":
        build_baseline(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        tok = spot_tokens(np.datetime64(sys.argv[1]), np.random.default_rng(0))
        assert tok.shape[1] == F_SPOT and np.isfinite(tok).all()
        if len(tok):
            nz = tok[:, 6:16][tok[:, 6:16] != 0]
            print(tok.shape, "act anomaly mean/std (nonzero):", nz.mean().round(3), nz.std().round(3), "| sources:", tok[:, 27:29].sum(0), "| dt range h:", (tok[:, 5] * 24).min().round(1), (tok[:, 5] * 24).max().round(1))
        else:
            print(tok.shape, "(no aggregates for this time)")
