"""Hourly WSPR aggregates on path midpoints: the candidate spot-derived input (PLAN.md Phase 3).

    uv run analysis/wspr_aggregate.py 2024-01 2024-12 /kass/forecast/eval/wspr_hourly_2024.parquet [--source wspr|pskreporter]

pskreporter: daily files, FT8 only (mode column), no power column; otherwise the same aggregate.

Per (hour, 5° midpoint cell, HF band): spot count on 1000-3000 km paths (single F2 hop), median SNR,
distinct tx and rx. Bands are the integer MHz codes in wspr.rx (1,3,5,7,10,14,18,21,24,28).
"""
import sys
from pathlib import Path

import duckdb

HF = (1, 3, 5, 7, 10, 14, 18, 21, 24, 28)


def month_files(m0, m1, source):
    y, m = map(int, m0.split("-")); out = []
    while f"{y:04d}-{m:02d}" <= m1:
        if source == "wspr":
            out.append(Path(f"/kass/forecast/spots/wspr/{y:04d}-{m:02d}.parquet"))
        else:
            out += sorted(Path("/kass/forecast/spots/pskreporter").glob(f"{y:04d}-{m:02d}-*.parquet"))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return [f for f in out if f.exists()]


CP_KM = 1500  # control points: this far along the great circle from each end of a >3000 km path (first/last F2 hop)


def _dest(lat1, lon1, lat2, lon2, km):
    """SQL for the point `km` along the great circle from (lat1,lon1) towards (lat2,lon2): (lat_expr, lon_expr) in degrees."""
    d = f"({km} / 6371.0)"
    brg = (f"atan2(sin(radians({lon2}) - radians({lon1})) * cos(radians({lat2})), "
           f"cos(radians({lat1})) * sin(radians({lat2})) - sin(radians({lat1})) * cos(radians({lat2})) * cos(radians({lon2}) - radians({lon1})))")
    lat = f"asin(sin(radians({lat1})) * cos({d}) + cos(radians({lat1})) * sin({d}) * cos({brg}))"
    lon = f"radians({lon1}) + atan2(sin({brg}) * sin({d}) * cos(radians({lat1})), cos({d}) - sin(radians({lat1})) * sin({lat}))"
    return f"degrees({lat})", f"degrees(atan2(sin({lon}), cos({lon})))"


def aggregate(con, files, out):
    r = RES_MIN[0]
    BIN = "date_trunc('hour', time::TIMESTAMP)" if r == 60 else f"to_timestamp(floor(epoch(time::TIMESTAMP) / {r * 60}) * {r * 60})::TIMESTAMP"
    base = f"""FROM read_parquet([{files}])
            WHERE band IN ({", ".join(map(str, HF))}) {MODE_FILTER[0]} {RX_FILTER[0]}
              AND NOT (tx_lat = 0 AND tx_lon = 0) AND NOT (rx_lat = 0 AND rx_lon = 0)"""
    if CONTROL_POINTS[0]:  # each long-path spot contributes at two control points, one near each end
        a_lat, a_lon = _dest("tx_lat", "tx_lon", "rx_lat", "rx_lon", CP_KM); b_lat, b_lon = _dest("rx_lat", "rx_lon", "tx_lat", "tx_lon", CP_KM)
        src = f"""SELECT {BIN} AS hour, band, {a_lat} AS mlat, {a_lon} AS mlon, snr, tx_sign, rx_sign {base} AND distance > 3000
                  UNION ALL
                  SELECT {BIN} AS hour, band, {b_lat} AS mlat, {b_lon} AS mlon, snr, tx_sign, rx_sign {base} AND distance > 3000"""
    else:
        src = f"""SELECT {BIN} AS hour, band,
                   -- midpoint of the great-circle path (good enough at <=3000 km: mean in lat, circular mean in lon)
                   (tx_lat + rx_lat) / 2 AS mlat,
                   degrees(atan2(sin(radians(tx_lon)) + sin(radians(rx_lon)), cos(radians(tx_lon)) + cos(radians(rx_lon)))) AS mlon,
                   snr, tx_sign, rx_sign {base} AND distance BETWEEN 1000 AND 3000"""
    con.execute(f"""
        COPY (
          WITH s AS ({src})
          SELECT hour, floor(mlat / 5)::INT * 5 AS cell_lat, floor(mlon / 5)::INT * 5 AS cell_lon, band,
                 count(*) AS n, median(snr) AS snr_med, sum(snr)::DOUBLE AS snr_sum, count(DISTINCT tx_sign) AS n_tx, count(DISTINCT rx_sign) AS n_rx
          FROM s GROUP BY 1, 2, 3, 4
        ) TO '{out}' (FORMAT parquet, COMPRESSION zstd)""")


MODE_FILTER = [""]
THREADS = [6]
MEMORY = ["48GB"]  # --memory 96GB: a month of FT8 control points at 5-min bins needs more than 48 GB
RES_MIN = [60]  # --res 5: 5-minute bins (spot_tokens then forms hour bins ending at T-15 min instead of clock hours)
CONTROL_POINTS = [False]  # --control-points: >3000 km paths at 1500 km from each end instead of 1000-3000 km midpoints
RX_FILTER = [""]  # --rx lat,lon: only spots received at that location (single-receiver / standalone-app study)


def main(m0, m1, out, source="wspr"):
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEMORY[0]}'; SET threads={THREADS[0]}; SET TimeZone='UTC'; SET preserve_insertion_order=false")
    MODE_FILTER[0] = "AND mode = 'FT8'" if source == "pskreporter" else ""
    tmp = Path(out).with_suffix(".parts"); tmp.mkdir(exist_ok=True)
    y, m = map(int, m0.split("-"))
    parts = []
    while f"{y:04d}-{m:02d}" <= m1:  # one month at a time: a year of FT8 (~8B rows) does not fit in memory
        month = f"{y:04d}-{m:02d}"
        files = ", ".join(f"'{f}'" for f in month_files(month, month, source))
        part = tmp / f"{month}.parquet"
        if files and not part.exists():
            aggregate(con, files, part)
            print(month, flush=True)
        if part.exists():
            parts.append(part)
        m += 1
        if m == 13:
            y, m = y + 1, 1
    con.execute(f"COPY (SELECT * FROM read_parquet({[str(p) for p in parts]})) TO '{out}' (FORMAT parquet, COMPRESSION zstd)")
    print(con.execute(f"SELECT count(*), count(DISTINCT hour), sum(n) FROM '{out}'").fetchone())


if __name__ == "__main__":
    src = sys.argv[sys.argv.index("--source") + 1] if "--source" in sys.argv else "wspr"
    CONTROL_POINTS[0] = "--control-points" in sys.argv
    if "--res" in sys.argv:
        RES_MIN[0] = int(sys.argv[sys.argv.index("--res") + 1])
    if "--memory" in sys.argv:
        MEMORY[0] = sys.argv[sys.argv.index("--memory") + 1]
    if "--threads" in sys.argv:
        THREADS[0] = int(sys.argv[sys.argv.index("--threads") + 1])
    if "--rx" in sys.argv:
        lat, lon = map(float, sys.argv[sys.argv.index("--rx") + 1].split(","))
        RX_FILTER[0] = f"AND abs(rx_lat - ({lat})) < 0.01 AND abs(rx_lon - ({lon})) < 0.01"
    main(sys.argv[1], sys.argv[2], sys.argv[3], src)
