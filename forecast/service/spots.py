"""Live spot loader (PLAN.md 2026-09-08): incremental WSPR / pskreporter pulls from the public ClickHouse HTTP
endpoint into local parquet, then the trailing-26 h hourly aggregates the forecast service's spot tokens read.

    python service/spots.py            # every 15 min (prop-spots.timer, :08/:23/:38/:53, with the ionosonde fetch)

Layout under $SPOTS_DIR (default /checkpoints/spots, the bind mount the service already has):
  raw/{wspr,psk}/<fetch time>.parquet   one file per pull, HF bands and the aggregate's columns only; 3 days kept
  agg/{wspr,psk,wspr_cp,psk_cp}_hourly_live.parquet   complete hours in [now-26h, now), same schema and code path
                                                       as the training aggregates (analysis/wspr_aggregate.aggregate)
  watermark.json                        last fetched `time` per source
The service sets SPOT_AGG_DIR=$SPOTS_DIR/agg and SPOT_BASELINE=$SPOTS_DIR/spot_baseline_cp.parquet (shipped once).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import wspr_aggregate as agg  # noqa: E402

ENDPOINT = os.environ.get("SPOTS_ENDPOINT", "https://wd1.wsprdaemon.org/")
ROOT = Path(os.environ.get("SPOTS_DIR", "/checkpoints/spots"))
TABLES = {"wspr": "wspr.rx", "psk": "pskreporter.rx"}
COLS = {"wspr": "id, time, band, rx_sign, rx_lat, rx_lon, tx_sign, tx_lat, tx_lon, distance, snr",
        "psk": "time, band, mode, rx_sign, rx_lat, rx_lon, tx_sign, tx_lat, tx_lon, distance, snr"}
KEY = {"wspr": "id", "psk": "time, band, mode, rx_sign, tx_sign"}
OVERLAP = timedelta(minutes=10); WINDOW_H = 26; RETAIN_DAYS = 3
RES = int(os.environ.get("SPOT_RES", 60)); TAG = "hourly" if RES == 60 else f"{RES}min"  # must match the checkpoint's samples


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


if __name__ == "__main__":
    main()
