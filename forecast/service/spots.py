"""Live spot loader (PLAN.md 2026-09-08): incremental WSPR / pskreporter pulls from the public ClickHouse HTTP
endpoint into local parquet, then the trailing-26 h hourly aggregates the forecast service's spot tokens read.

    python service/spots.py            # every 15 min (prop-spots.timer, :08/:23/:38/:53, with the ionosonde fetch)

Layout under $SPOTS_DIR (default /checkpoints/spots, the bind mount the service already has):
  raw/{wspr,psk}/<fetch time>.parquet   one file per pull, HF bands and the aggregate's columns only; 3 days kept
  agg/{wspr,psk,wspr_cp,psk_cp}_hourly_live.parquet   complete hours in [now-26h, now), same schema and code path
                                                       as the training aggregates (analysis/wspr_aggregate.aggregate)
  agg/hist/{src}_hourly_YYYY-MM.parquet the same aggregates accumulated per calendar month, HIST_MONTHS kept;
                                        what the rolling baseline is built from (~300-500 MB for 3 months)
  spot_baseline_live.parquet            rolling baseline for the current month, rebuilt when the month turns
  watermark.json                        last fetched `time` per source
The service sets SPOT_AGG_DIR=$SPOTS_DIR/agg and SPOT_BASELINE to the baseline the running checkpoint was trained
against: the shipped spot_baseline_cp.parquet for a frozen-baseline checkpoint, spot_baseline_live.parquet for a
rolling one (app.py refuses to serve a checkpoint against the other scheme).
Rolling baselines are causal, so the one for month M is built from the BASELINE_WINDOW months before M and is
final the moment M begins: one rebuild a month, and the step it puts in the input distribution is one month of
traffic drift (~0.005 in log10) rather than the years a manual rebuild of a frozen baseline would move it.
Cold start: until HIST_MONTHS of history have accumulated the rebuild is skipped and whatever baseline is in
place keeps being used, so a fresh host is seeded by shipping one alongside the checkpoint.
Nothing here touches /kass: live serving is local-disk only.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "analysis")); sys.path.insert(0, str(ROOT_DIR / "train"))
import wspr_aggregate as agg  # noqa: E402

ENDPOINT = os.environ.get("SPOTS_ENDPOINT", "https://wd1.wsprdaemon.org/")
ROOT = Path(os.environ.get("SPOTS_DIR", "/checkpoints/spots"))
TABLES = {"wspr": "wspr.rx", "psk": "pskreporter.rx"}
COLS = {"wspr": "id, time, band, rx_sign, rx_lat, rx_lon, tx_sign, tx_lat, tx_lon, distance, snr",
        "psk": "time, band, mode, rx_sign, rx_lat, rx_lon, tx_sign, tx_lat, tx_lon, distance, snr"}
KEY = {"wspr": "id", "psk": "time, band, mode, rx_sign, tx_sign"}
OVERLAP = timedelta(minutes=10); WINDOW_H = 26; RETAIN_DAYS = 3
RES = int(os.environ.get("SPOT_RES", 60)); TAG = "hourly" if RES == 60 else f"{RES}min"  # must match the checkpoint's samples
HIST_MONTHS = int(os.environ.get("SPOT_HIST_MONTHS", 3))  # month files kept; must exceed spot_tokens.BASELINE_WINDOW


def fetch(sql):
    with urllib.request.urlopen(ENDPOINT + "?query=" + urllib.parse.quote(sql + " FORMAT Parquet"), timeout=120) as r:
        return r.read()


def pull(src, wm):
    """Rows with time > wm - overlap (dedupe happens at aggregation). Returns (n_rows, max_time)."""
    wm = wm.astimezone(timezone.utc)
    sql = f"""SELECT {COLS[src]} FROM {TABLES[src]} WHERE time > toDateTime('{wm:%Y-%m-%d %H:%M:%S}', 'UTC')
              AND band IN ({", ".join(map(str, agg.HF))}) {"AND mode = 'FT8'" if src == "psk" else ""}"""
    data = fetch(sql)
    out = ROOT / "raw" / src / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True); out.write_bytes(data)
    con = duckdb.connect(); con.execute("SET TimeZone='UTC'")
    n, mx = con.execute(f"SELECT count(*), max(time) FROM '{out}'").fetchone()
    if n == 0:
        out.unlink()
    return n, None if mx is None else (mx if mx.tzinfo else mx.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def aggregate(src, now):
    files = sorted((ROOT / "raw" / src).glob("*.parquet"))
    if not files:
        return 0
    # duckdb spills the row_number() sort to disk once the window outgrows memory_limit; its default spill dir is
    # ./.tmp, which is /app (read-only) in the pod. Spill under SPOTS_DIR instead and give it more room.
    (ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(); con.execute(f"SET memory_limit='8GB'; SET threads=2; SET TimeZone='UTC'; SET temp_directory='{ROOT / 'tmp'}'")
    t0 = now - timedelta(hours=WINDOW_H); t1 = now.replace(minute=now.minute - now.minute % RES, second=0, microsecond=0)  # complete bins only
    dedup = ROOT / "agg" / f"_{src}_window.parquet"; dedup.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"""COPY (SELECT * FROM (SELECT *, row_number() OVER (PARTITION BY {KEY[src]}) AS _r
                    FROM read_parquet({[str(f) for f in files]}, union_by_name=true)
                    WHERE time >= '{t0:%Y-%m-%d %H:%M:%S}' AND time < '{t1:%Y-%m-%d %H:%M:%S}') WHERE _r = 1)
                    TO '{dedup}' (FORMAT parquet)""")
    n = con.execute(f"SELECT count(*) FROM '{dedup}'").fetchone()[0]
    agg.MODE_FILTER[0] = ""; agg.RES_MIN[0] = RES  # raw files are already FT8-only for psk
    for cp, name in ((False, src), (True, f"{src}_cp")):
        agg.CONTROL_POINTS[0] = cp
        tmp = ROOT / "agg" / f"_{name}.tmp.parquet"; agg.aggregate(con, f"'{dedup}'", tmp)
        os.replace(tmp, ROOT / "agg" / f"{name}_{TAG}_live.parquet")
    dedup.unlink()
    return n


def archive(con, now):
    """Accumulate the live window's complete bins into agg/hist/{src}_{TAG}_YYYY-MM.parquet, which is what the
    rolling baseline is built from. The live aggregate covers the trailing 26 h and is rewritten every run, so
    every bin is archived many times over; the newest copy wins, since late-arriving spots only raise a count.
    A gap longer than WINDOW_H is lost from the history for good - the baseline builder's support widening is
    what covers that, and `n_hours` in the baseline is what shows it happened."""
    hist = ROOT / "agg" / "hist"; hist.mkdir(parents=True, exist_ok=True)
    for name in list(TABLES) + [f"{s}_cp" for s in TABLES]:
        live = ROOT / "agg" / f"{name}_{TAG}_live.parquet"
        if not live.exists():
            continue
        for (mon,) in con.execute(f"SELECT DISTINCT date_trunc('month', hour) FROM '{live}' ORDER BY 1").fetchall():
            f = hist / f"{name}_{TAG}_{mon:%Y-%m}.parquet"
            keep = f"UNION ALL BY NAME SELECT *, 0 AS _new FROM '{f}'" if f.exists() else ""
            tmp = f.with_suffix(".tmp.parquet")
            con.execute(f"""COPY (SELECT * EXCLUDE (_new, _r) FROM (
                               SELECT *, row_number() OVER (PARTITION BY hour, cell_lat, cell_lon, band ORDER BY _new DESC) AS _r
                               FROM (SELECT *, 1 AS _new FROM '{live}' WHERE date_trunc('month', hour) = TIMESTAMP '{mon}' {keep}))
                             WHERE _r = 1) TO '{tmp}' (FORMAT parquet, COMPRESSION zstd)""")
            os.replace(tmp, f)
    cut = now.replace(day=1)
    for _ in range(HIST_MONTHS):
        cut = (cut - timedelta(days=1)).replace(day=1)
    cut = f"{cut:%Y-%m}"
    for f in hist.glob(f"*_{TAG}_*.parquet"):
        if f.stem.split("_")[-1] < cut:
            f.unlink()


def refresh_baseline(con, now):
    """Rebuild the rolling baseline when the month turns. The baseline for month M reads only months before M,
    so it is final on the 1st and never needs revisiting; if SPOT_BASELINE is a frozen baseline, or the history
    does not reach back far enough yet, leave it alone."""
    import spot_tokens as spot_mod  # imports the model stack (PyIRI, psycopg); keep it off the fetch path
    out = Path(os.environ.get("SPOT_BASELINE", ROOT / "spot_baseline_live.parquet"))
    if out.exists() and "for_month" not in {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM '{out}'").fetchall()}:
        return print(f"baseline: {out.name} is a frozen baseline, leaving it alone "
                     "(point SPOT_BASELINE at a rolling baseline to switch schemes)", flush=True)
    mon = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if out.exists() and con.execute(f"SELECT count(*) FROM '{out}' WHERE for_month = TIMESTAMP '{mon:%Y-%m-%d}'").fetchone()[0]:
        return
    hist = ROOT / "agg" / "hist"
    want = [(mon - timedelta(days=1)).replace(day=1)]
    while len(want) < spot_mod.BASELINE_WINDOW[0]:
        want.append((want[-1] - timedelta(days=1)).replace(day=1))
    missing = [m for m in want if not (hist / f"wspr_{TAG}_{m:%Y-%m}.parquet").exists()]
    if missing:
        return print(f"baseline: keeping the current one; no history yet for {[f'{m:%Y-%m}' for m in missing]}", flush=True)
    spot_mod.RES[0] = RES
    tmp = out.with_suffix(".tmp.parquet")
    spot_mod.build_rolling_baseline(tmp, f"{mon:%Y-%m}", f"{mon:%Y-%m}", agg_dir=hist, conn=con)
    os.replace(tmp, out)
    print(f"baseline: rebuilt {out} for {mon:%Y-%m} from {[f'{m:%Y-%m}' for m in want]}", flush=True)


def prune(src, now):
    for f in (ROOT / "raw" / src).glob("*.parquet"):
        if datetime.strptime(f.stem, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc) < now - timedelta(days=RETAIN_DAYS):
            f.unlink()


def main():
    now = datetime.now(timezone.utc); ROOT.mkdir(parents=True, exist_ok=True)
    wmf = ROOT / "watermark.json"; wm = json.loads(wmf.read_text()) if wmf.exists() else {}
    for src in TABLES:
        t = time.time()
        since = datetime.fromisoformat(wm[src]) if src in wm else now - timedelta(hours=WINDOW_H)
        n, mx = pull(src, since - OVERLAP)
        if mx is not None:
            wm[src] = mx.isoformat()
        wmf.write_text(json.dumps(wm))
        prune(src, now)
        nw = aggregate(src, now)
        print(f"{src}: pulled {n:,} rows since {since:%Y-%m-%d %H:%M} (max {mx}), window {nw:,} rows -> aggregates, {time.time() - t:.1f}s", flush=True)
    (ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(); con.execute(f"SET memory_limit='8GB'; SET threads=2; SET TimeZone='UTC'; SET temp_directory='{ROOT / 'tmp'}'")
    archive(con, now)
    refresh_baseline(con, now)


if __name__ == "__main__":
    main()
