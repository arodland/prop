"""Spot-activity tokens (PLAN.md Phase 4b): one token per (5° midpoint cell, hour, source) from the hourly
WSPR / FT8 aggregates, carrying per-band activity and SNR anomalies against a climatological baseline.

    uv run train/spot_tokens.py baseline /kass/forecast/eval/spot_baseline.parquet 2019 2023   # frozen baseline (legacy)
    uv run train/spot_tokens.py rolling /kass/forecast/eval/spot_baseline_roll.parquet 2019-01 2026-08   # rolling baseline
    uv run train/spot_tokens.py 2024-06-15T12:00                                               # smoke: token shape

Token features (F_SPOT = 5 + 1 + 10 + 10 + 1 + 2 = 29):
  lat/90, sin lon, cos lon, sin LT, cos LT | Δt/24 (hour end, negative) | activity anomaly × 10 bands |
  SNR anomaly × 10 bands (0 where the band is absent) | log1p(total spots)/8 | source one-hot (WSPR, FT8)
Activity anomaly = log1p(n) − baseline median. Two baseline schemes, picked automatically from the baseline
file’s columns (a `for_month` column means rolling), and recorded per sample so a checkpoint cannot be served
against the wrong one:
  frozen   keyed (cell, band, UT hour, calendar month), built once from fixed years — the original scheme.
           Its reference is tied to a build date, so the anomaly drifts with WSPR/FT8 traffic growth: measured
           +0.059 log10/yr, +0.19 by 2025 Q4, still rising (see spot-baseline-design.md).
  rolling  keyed (cell, band, UT hour, target month), the median over the BASELINE_WINDOW calendar months
           *preceding* that target month. Causal, so it cannot go stale, needs no recalculation schedule, is
           the same rule for WSPR and FT8 (the frozen builder exempted FT8, a leak), and self-calibrates to
           whatever feed it is given. Keys too thin in that window are rebuilt over BASELINE_WIDEN months;
           keys still missing fall back to the band’s global median for that UT hour.
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
BASELINE_WINDOW = [2]  # rolling baseline: calendar months before the target month; 2 is what spot-baseline-design.md measured
BASELINE_WIDEN = [6]   # keys with < MIN_HOURS in that window get a wider window instead of being dropped
MIN_HOURS = 6          # baseline support floor, same number the frozen scheme applied at token time
SCHEME = ["frozen"]    # set by con() from the baseline file's columns; build_samples records it in every sample
MEMORY = ["32GB"]      # build_rolling_baseline only; the token path runs in the 8 GB the service gives it
THREADS = [6]
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
            # materialised, not a view over the parquet: build_samples calls spot_tokens once per issue time
            # (~17k times) and a rolling baseline is ~8x the rows of a frozen one, so re-scanning the file every
            # call is the whole cost. `base` stays a view over it so the presence checks below still see it.
            _con.execute(f"CREATE TABLE _base_rows AS SELECT * FROM '{BASELINE}'")
            _con.execute("CREATE VIEW base AS SELECT * FROM _base_rows")
            SCHEME[0] = baseline_scheme()
    return _con


def baseline_scheme():
    """Which scheme the configured baseline file implements: 'rolling' if it is keyed by target month,
    else 'frozen'. Recorded in every sample and checked against the checkpoint at serving time."""
    if not BASELINE.exists():
        return SCHEME[0]
    cols = duckdb.connect().execute(f"DESCRIBE SELECT * FROM '{BASELINE}'").fetchall()
    return "rolling" if "for_month" in {r[0] for r in cols} else "frozen"


def _hourly_sql(files):
    """One source’s aggregate files as (cell, band, clock hour, n, snr); 5-min bins rolled up so the two
    resolutions share the baseline semantics (snr weighted by count, as spot_tokens does at token time)."""
    if RES[0] == 60:
        return f"SELECT cell_lat, cell_lon, band, hour, n, snr_med AS snr FROM read_parquet({files}, union_by_name=true)"
    return f"""SELECT cell_lat, cell_lon, band, date_trunc('hour', hour) AS hour, sum(n) AS n, sum(snr_sum) / sum(n) AS snr
               FROM read_parquet({files}, union_by_name=true) GROUP BY 1, 2, 3, 4"""


def build_baseline(out, y0, y1):
    """Frozen baseline from years y0..y1 for WSPR (training years only, so eval years never see their own statistics).
    FT8 only exists from 2024-12, so its baseline uses all its years regardless (a mild, accepted leak).
    Superseded by build_rolling_baseline — kept because every checkpoint through v9b was trained against it."""
    c = duckdb.connect(); c.execute("SET memory_limit='24GB'; SET TimeZone='UTC'")
    parts = []
    for src in list(SOURCES) + [f"{s}_cp" for s in SOURCES]:
        tag = "hourly" if RES[0] == 60 else f"{RES[0]}min"
        files = [str(f) for f in sorted(AGG.glob(f"{src}_{tag}_*.parquet")) if src.startswith("psk") or y0 <= int(f.stem.split('_')[-1]) <= y1]
        if not files:
            continue
        hourly = _hourly_sql(files)
        parts.append(f"""SELECT source, cell_lat, cell_lon, band, ut, month, median(logn) AS med_logn, median(snr) AS med_snr, count(*) AS n_hours
            FROM (SELECT '{src}' AS source, cell_lat, cell_lon, band, extract(hour FROM a.hour)::INT AS ut, extract(month FROM a.hour)::INT AS month, log(1 + n) AS logn, snr
                  FROM ({hourly}) a)
            GROUP BY source, cell_lat, cell_lon, band, ut, month""")
    c.execute(f"COPY ({' UNION ALL '.join(parts)}) TO '{out}' (FORMAT parquet)")
    print(c.execute(f"SELECT source, count(*) FROM '{out}' GROUP BY 1").fetchall())


def build_rolling_baseline(out, m0, m1, agg_dir=None, window=None, widen=None, conn=None):
    """Causal baseline (spot-baseline-design.md): for each target month in m0..m1 ("YYYY-MM"), the median of
    log10(1+n) over the `window` calendar months *before* it, keyed (source, cell, band, UT hour, target month).
    The same rule for every source — nothing is exempt and nothing reaches forward of its own target month, so
    there is no year range to choose and no leak to accept. Rows are emitted in three tiers:
      1. keys with >= MIN_HOURS in the window;
      2. keys too thin there but with >= MIN_HOURS over `widen` months, remeasured over that wider window;
      3. one cell_lat IS NULL row per (source, band, UT, month) — the median over the cells above — which
         spot_tokens falls back to, so a newly active cell yields a token instead of vanishing.
    Cost is one pass per source with the window join; `conn` lets a caller (service/spots.py) pass its own."""
    agg_dir = Path(agg_dir) if agg_dir else AGG
    w, wide = window or BASELINE_WINDOW[0], widen or BASELINE_WIDEN[0]
    c = conn
    if c is None:
        c = duckdb.connect()
        c.execute(f"SET memory_limit='{MEMORY[0]}'; SET threads={THREADS[0]}; SET TimeZone='UTC'; SET preserve_insertion_order=false")
    tag = "hourly" if RES[0] == 60 else f"{RES[0]}min"
    c.execute(f"""CREATE OR REPLACE TEMP TABLE tgt AS
        SELECT unnest(generate_series(DATE '{m0}-01', DATE '{m1}-01', INTERVAL 1 MONTH))::TIMESTAMP AS mon""")
    c.execute("""CREATE OR REPLACE TEMP TABLE out_rows (source VARCHAR, cell_lat INT, cell_lon INT, band INT,
                 ut INT, for_month TIMESTAMP, med_logn DOUBLE, med_snr DOUBLE, n_hours BIGINT)""")
    for src in list(SOURCES) + [f"{s}_cp" for s in SOURCES]:
        files = [str(f) for f in sorted(agg_dir.glob(f"{src}_{tag}_*.parquet"))]
        if not files:
            continue
        c.execute(f"""CREATE OR REPLACE TEMP TABLE h AS
            SELECT cell_lat, cell_lon, band, extract(hour FROM hour)::INT AS ut, date_trunc('month', hour) AS mon,
                   log(1 + n) AS logn, snr
            FROM ({_hourly_sql(files)})""")
        c.execute(f"""CREATE OR REPLACE TEMP TABLE narrow AS
            SELECT h.cell_lat, h.cell_lon, h.band, h.ut, t.mon AS for_month,
                   median(h.logn) AS med_logn, median(h.snr) AS med_snr, count(*) AS n_hours
            FROM h JOIN tgt t ON h.mon < t.mon AND h.mon >= t.mon - INTERVAL {w} MONTH
            GROUP BY 1, 2, 3, 4, 5""")
        # which keys need the wider window: enumerated from a per-(key, month) count table, which is small
        # enough to expand over `widen` months where the hourly rows are not
        c.execute(f"""CREATE OR REPLACE TEMP TABLE thin AS
            SELECT wd.cell_lat, wd.cell_lon, wd.band, wd.ut, wd.for_month FROM (
                SELECT k.cell_lat, k.cell_lon, k.band, k.ut, t.mon AS for_month, sum(k.n) AS n_wide
                FROM (SELECT cell_lat, cell_lon, band, ut, mon, count(*) AS n FROM h GROUP BY 1, 2, 3, 4, 5) k
                JOIN tgt t ON k.mon < t.mon AND k.mon >= t.mon - INTERVAL {wide} MONTH
                GROUP BY 1, 2, 3, 4, 5) wd
            LEFT JOIN narrow b ON b.cell_lat = wd.cell_lat AND b.cell_lon = wd.cell_lon AND b.band = wd.band
                 AND b.ut = wd.ut AND b.for_month = wd.for_month
            WHERE wd.n_wide >= {MIN_HOURS} AND coalesce(b.n_hours, 0) < {MIN_HOURS}""")
        c.execute(f"""CREATE OR REPLACE TEMP TABLE cells AS
            SELECT * FROM narrow WHERE n_hours >= {MIN_HOURS}
            UNION ALL
            SELECT k.cell_lat, k.cell_lon, k.band, k.ut, k.for_month,
                   median(h.logn) AS med_logn, median(h.snr) AS med_snr, count(*) AS n_hours
            FROM thin k JOIN h ON h.cell_lat = k.cell_lat AND h.cell_lon = k.cell_lon AND h.band = k.band
                 AND h.ut = k.ut AND h.mon < k.for_month AND h.mon >= k.for_month - INTERVAL {wide} MONTH
            GROUP BY 1, 2, 3, 4, 5""")
        c.execute(f"""INSERT INTO out_rows
            SELECT '{src}', cell_lat, cell_lon, band, ut, for_month, med_logn, med_snr, n_hours FROM cells
            UNION ALL
            SELECT '{src}', NULL::INT, NULL::INT, band, ut, for_month,
                   median(med_logn), median(med_snr), sum(n_hours)
            FROM cells GROUP BY band, ut, for_month""")
        print(f"  {src}: {c.execute('SELECT count(*) FROM cells').fetchone()[0]:,} cell rows, "
              f"{c.execute('SELECT count(*) FROM thin').fetchone()[0]:,} of them widened to {wide} months", flush=True)
    c.execute(f"COPY out_rows TO '{out}' (FORMAT parquet, COMPRESSION zstd)")
    print(f"wrote {out}, window {w} months:",
          c.execute(f"SELECT source, count(*), count(*) FILTER (cell_lat IS NULL) AS fallback_rows FROM '{out}' GROUP BY 1 ORDER BY 1").fetchall())


def _baseline_lookup(view, h):
    """(med_logn expr, med_snr expr, join clause, extra WHERE) for the active scheme. `h` is the SQL expression
    for the hour the baseline is keyed on: the clock hour at res 60, the bin midpoint at res 5."""
    if SCHEME[0] != "rolling":  # frozen: inner join, support floor applied here, cells absent from it are dropped
        return ("b.med_logn", "b.med_snr",
                f"JOIN base b ON b.source = '{view}' AND b.cell_lat = a.cell_lat AND b.cell_lon = a.cell_lon AND b.band = a.band"
                f" AND b.ut = extract(hour FROM {h})::INT AND b.month = extract(month FROM {h})::INT",
                f"b.n_hours >= {MIN_HOURS}")
    # rolling: the support floor is already applied by the builder, so this is a left join onto the target
    # month's rows, falling back to the band's global row (cell_lat IS NULL) for a cell the window never saw
    def on(al):
        return (f"{al}.source = '{view}' AND {al}.band = a.band AND {al}.ut = extract(hour FROM {h})::INT"
                f" AND {al}.for_month = date_trunc('month', {h})")
    return ("coalesce(b.med_logn, g.med_logn)", "coalesce(b.med_snr, g.med_snr)",
            f"LEFT JOIN base b ON {on('b')} AND b.cell_lat = a.cell_lat AND b.cell_lon = a.cell_lon"
            f" LEFT JOIN base g ON {on('g')} AND g.cell_lat IS NULL",
            "coalesce(b.med_logn, g.med_logn) IS NOT NULL")


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
                med, msnr, join, filt = _baseline_lookup(view, "a.hour")
                df = c.execute(f"""
                    SELECT a.hour, a.cell_lat, a.cell_lon, a.band, log(1 + a.n) - {med} AS act, coalesce(a.snr_med - {msnr}, 0) AS snr, a.n
                    FROM {view} a {join}
                    WHERE a.hour >= ? AND a.hour <= ? AND {filt}""", [t0 - pd.Timedelta(hours=24), t0 - pd.Timedelta(hours=2)]).df()
            else:
                # bins k = 0..23 end at E0 - k h, E0 = floor_res(T - LAG_MIN); `hour` below is the bin *start* so the
                # downstream code (hour_end = hour + 1 h) is shared; baseline keyed by the UT hour of the bin midpoint
                e0 = (t0 - pd.Timedelta(minutes=LAG_MIN)).floor(f"{RES[0]}min")
                med, msnr, join, filt = _baseline_lookup(view, "a.mid")
                df = c.execute(f"""
                    WITH bins AS (
                      SELECT cell_lat, cell_lon, band, floor(epoch(? - hour) / 3600 - 1e-9)::INT AS k, sum(n) AS n, sum(snr_sum) / sum(n) AS snr
                      FROM {view} WHERE hour >= ? AND hour < ? GROUP BY 1, 2, 3, 4)
                    SELECT a.hour, a.cell_lat, a.cell_lon, a.band, log(1 + a.n) - {med} AS act, coalesce(a.snr - {msnr}, 0) AS snr, a.n
                    FROM (SELECT *, ? - INTERVAL (k + 1) HOUR AS hour, ? - INTERVAL (k) HOUR - INTERVAL 30 MINUTE AS mid FROM bins) a {join}
                    WHERE {filt}""", [e0, e0 - pd.Timedelta(hours=24), e0, e0, e0]).df()
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
    if "--res" in sys.argv:
        RES[0] = int(sys.argv[sys.argv.index("--res") + 1])
    if sys.argv[1] == "baseline":
        build_baseline(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif sys.argv[1] == "rolling":  # out m0 m1 [window] [widen]; months as YYYY-MM
        build_rolling_baseline(sys.argv[2], sys.argv[3], sys.argv[4],
                               window=int(sys.argv[5]) if len(sys.argv) > 5 and not sys.argv[5].startswith("--") else None,
                               widen=int(sys.argv[6]) if len(sys.argv) > 6 and not sys.argv[6].startswith("--") else None)
    else:
        tok = spot_tokens(np.datetime64(sys.argv[1]), np.random.default_rng(0))
        assert tok.shape[1] == F_SPOT and np.isfinite(tok).all()
        if len(tok):
            nz = tok[:, 6:16][tok[:, 6:16] != 0]
            print(tok.shape, f"baseline {SCHEME[0]}", "| act anomaly mean/std (nonzero):", nz.mean().round(3), nz.std().round(3), "| sources:", tok[:, 27:29].sum(0), "| dt range h:", (tok[:, 5] * 24).min().round(1), (tok[:, 5] * 24).max().round(1))
        else:
            print(tok.shape, "(no aggregates for this time)")
