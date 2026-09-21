"""Phase 3 check for geomagnetic/solar indices: does Kp/ap at T (known at T) predict the station foF2 anomaly at T+h
beyond the station's own anomaly at T?

    uv run analysis/indices_value.py 2024-01-01 2025-01-01
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.iri_cache import iri_points_cached  # noqa: E402
from data.drivers import attach_indices, f107_trailing  # noqa: E402
from data.load import SNAPSHOT, connect  # noqa: E402


def main(start, end):
    con = connect(); con.execute("SET memory_limit='32GB'; SET TimeZone='UTC'"); attach_indices(con, SNAPSHOT)
    st = con.execute(f"""
      SELECT i.station_id, date_trunc('hour', i.time) AS hour, avg(i.fof2) AS fof2, any_value(s.lat) AS lat, any_value(s.lon) AS lon
      FROM iono i JOIN station s ON s.id=i.station_id WHERE i.time >= '{start}' AND i.time < '{end}' AND i.fof2 IS NOT NULL GROUP BY 1,2""").df()
    st["time"] = st.hour + pd.Timedelta(minutes=30); out = np.full(len(st), np.nan)
    for day, idx in st.groupby(st.time.dt.floor("D")).indices.items():
        out[idx] = iri_points_cached(st.time.to_numpy()[idx], st.lat.to_numpy()[idx], st.lon.to_numpy()[idx], f107_trailing(con, str(day)))["fof2"]
    st["a_obs"] = st.fof2 - out
    # 3-hourly Kp/ap known at T: the slot containing T (GFZ final values; real-time Kp is a nowcast, close enough for a value check)
    d = con.execute("SELECT date, kp3h, ap3h, f107_obs FROM daily").df()
    kp = pd.DataFrame({"hour": np.repeat(pd.to_datetime(d.date), 8) + pd.to_timedelta(np.tile(np.arange(8) * 3, len(d)), unit="h"),
                       "kp": np.concatenate(d.kp3h.to_list()).astype(float), "ap": np.concatenate(d.ap3h.to_list()).astype(float)})
    kp = kp.set_index("hour").sort_index()
    st["slot"] = st.hour.dt.floor("3h")
    st = st.merge(kp, left_on="slot", right_index=True, how="left")
    st["ap_lag6"] = st.slot.map(kp.ap.shift(2))  # ap two slots earlier (storm onset history)
    st["ap_max24"] = st.slot.map(kp.ap.rolling(8, min_periods=1).max())
    print(f"# {len(st):,} station-hours {start}..{end}, {st.station_id.nunique()} stations")
    print("\n## R^2 of station foF2 anomaly at T+h from: own anomaly at T | own + ap(T), ap_max24(T), |lat|*ap | gain")
    s2 = st.set_index(["station_id", "hour"]).sort_index()
    for h in (1, 3, 6, 12, 24):
        fut = s2[["a_obs"]].rename(columns={"a_obs": "y"}).copy()
        fut.index = fut.index.set_levels(fut.index.levels[1] - pd.Timedelta(hours=h), level=1)
        j = s2[["a_obs", "ap", "ap_max24", "lat"]].join(fut, how="inner").dropna()
        y = j.y.to_numpy(); one = np.ones(len(j))
        def r2(X):
            b, *_ = np.linalg.lstsq(X, y, rcond=None); return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        a = r2(np.c_[one, j.a_obs]); c = r2(np.c_[one, j.a_obs, np.log1p(j.ap), np.log1p(j.ap_max24), np.abs(j.lat) * np.log1p(j.ap_max24)])
        print(f"   h={h:2d}: n={len(j):,}  own {a:.3f}  own+ap {c:.3f}  gain {c - a:+.3f}")
    print("\n## anomaly by ap class (all hours): mean foF2 anomaly, RMSE of IRI, by |lat| band")
    st["apc"] = pd.cut(st.ap_max24, [-1, 15, 40, 80, 1000], labels=["quiet", "unsettled", "active", "storm"])
    st["band"] = pd.cut(st.lat.abs(), [0, 30, 50, 90], labels=["<30", "30-50", ">50"])
    print(st.groupby(["band", "apc"], observed=True).a_obs.agg(n="size", mean="mean", rms=lambda x: np.sqrt((x ** 2).mean())).round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:3])
