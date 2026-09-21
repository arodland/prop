"""Phase 3 spot check, activity-anomaly formulation: per-band log-count anomaly vs the station MUF(3000) anomaly.

    uv run analysis/spot_activity_value.py /kass/forecast/eval/wspr_hourly_2024.parquet 2024-01-01 2025-01-01

Activity anomaly = log1p(n) minus the median for the same (cell, band, UT hour) within the month, i.e.
"is this band busier than usual here at this time of day". Not censored by band occupancy.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.iri_cache import iri_points_cached  # noqa: E402
from data.drivers import attach_indices, f107_trailing  # noqa: E402
from data.load import SNAPSHOT, connect  # noqa: E402


def main(agg, start, end):
    con = connect(); con.execute("SET memory_limit='32GB'; SET TimeZone='UTC'"); attach_indices(con, SNAPSHOT)
    con.execute(f"CREATE VIEW w AS SELECT * FROM '{agg}'")
    con.execute("""CREATE TABLE act AS
      SELECT hour, cell_lat, cell_lon, band, n, snr_med,
             log(1 + n) - median(log(1 + n)) OVER (PARTITION BY cell_lat, cell_lon, band, extract(hour FROM hour), date_trunc('month', hour)) AS act_anom,
             snr_med - median(snr_med) OVER (PARTITION BY cell_lat, cell_lon, band, extract(hour FROM hour), date_trunc('month', hour)) AS snr_anom
      FROM w""")
    st = con.execute(f"""
      WITH i AS (SELECT i.station_id, date_trunc('hour', i.time) AS hour, avg(i.mufd) AS mufd, any_value(s.lat) AS lat, any_value(s.lon) AS lon
                 FROM iono i JOIN station s ON s.id=i.station_id WHERE i.time >= '{start}' AND i.time < '{end}' AND i.mufd IS NOT NULL GROUP BY 1,2)
      SELECT i.*, a.band, a.n, a.act_anom, a.snr_anom
      FROM i JOIN act a ON a.hour = i.hour AND a.cell_lat = floor(i.lat/5)::INT*5 AND a.cell_lon = floor(i.lon/5)::INT*5""").df()
    base = st.drop_duplicates(["station_id", "hour"])[["station_id", "hour", "mufd", "lat", "lon"]].copy()
    base["time"] = base.hour + pd.Timedelta(minutes=30); out = np.full(len(base), np.nan)
    for day, idx in base.groupby(base.time.dt.floor("D")).indices.items():
        out[idx] = iri_points_cached(base.time.to_numpy()[idx], base.lat.to_numpy()[idx], base.lon.to_numpy()[idx], f107_trailing(con, str(day)))["mufd"]
    base["a_obs"] = base.mufd - out
    st = st.merge(base[["station_id", "hour", "a_obs"]], on=["station_id", "hour"])
    print(f"# {agg}  {start}..{end}: {len(base):,} station-hours with spot data in their cell, {base.station_id.nunique()} stations")
    print("\n## per-band correlation of activity / SNR anomaly with the station MUF(3000) anomaly (same cell-hour)")
    rows = []
    for b, d in st.groupby("band"):
        if len(d) < 2000:
            continue
        rows.append({"band": b, "n": len(d), "act_corr": np.corrcoef(d.act_anom, d.a_obs)[0, 1], "snr_corr": np.corrcoef(d.snr_anom.fillna(0), d.a_obs)[0, 1]})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    piv = st.pivot_table(index=["station_id", "hour"], columns="band", values=["act_anom", "snr_anom"]).fillna(0)
    y = base.set_index(["station_id", "hour"]).a_obs.reindex(piv.index).to_numpy()
    X = np.c_[np.ones(len(piv)), piv.to_numpy()]; m = np.isfinite(y); Xm, ym = X[m], y[m]
    months = piv.index.get_level_values("hour").month.to_numpy()[m]; sse = sst = 0.0
    for mo in np.unique(months):
        tr, te = months != mo, months == mo
        b, *_ = np.linalg.lstsq(Xm[tr], ym[tr], rcond=None); sse += ((ym[te] - Xm[te] @ b) ** 2).sum(); sst += ((ym[te] - ym[tr].mean()) ** 2).sum()
    b, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
    print(f"\n## all bands, activity + SNR anomalies -> same-hour MUF anomaly: n={m.sum():,}  R^2 in-sample {1 - ((ym - Xm @ b) ** 2).sum() / ((ym - ym.mean()) ** 2).sum():.3f}, leave-one-month-out {1 - sse / sst:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
