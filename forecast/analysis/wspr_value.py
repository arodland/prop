"""Phase 3 source-value check for WSPR: does a spot-derived MUF proxy carry information about the MUF(3000) anomaly?

    uv run analysis/wspr_value.py /kass/forecast/eval/wspr_hourly_2024.parquet [start end]

Proxy per (hour, 5° cell): highest HF band with >= K spots on 1000-3000 km paths (censored from above by
band occupancy: a band with spots is open, a band without may be closed or just unused). Compared with
ionosonde MUF(3000) at stations in the same cell, same hour, and with the anomaly relative to IRI MUF.
Also: does the proxy predict the station MUF anomaly at T+h beyond the station's own anomaly at T?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.iri_cache import iri_points_cached  # noqa: E402
from data.drivers import attach_indices, f107_trailing  # noqa: E402
from data.load import SNAPSHOT, connect  # noqa: E402

K = 3


def main(agg, start="2024-01-01", end="2025-01-01"):
    con = connect(); con.execute("SET memory_limit='24GB'; SET TimeZone='UTC'")
    attach_indices(con, SNAPSHOT)
    con.execute(f"CREATE VIEW w AS SELECT * FROM '{agg}'")
    con.execute(f"""CREATE TABLE proxy AS
        SELECT hour, cell_lat, cell_lon, max(CASE WHEN n >= {K} THEN band END) AS muf_proxy, sum(n) AS n_spots,
               max(CASE WHEN n >= {K} AND band >= 14 THEN snr_med END) AS snr_hi
        FROM w GROUP BY 1, 2, 3""")
    st = con.execute(f"""
        WITH i AS (SELECT i.station_id, date_trunc('hour', i.time) AS hour, avg(i.mufd) AS mufd, avg(i.fof2) AS fof2, any_value(s.lat) AS lat, any_value(s.lon) AS lon
                   FROM iono i JOIN station s ON s.id = i.station_id
                   WHERE i.time >= '{start}' AND i.time < '{end}' AND i.mufd IS NOT NULL GROUP BY 1, 2)
        SELECT i.*, p.muf_proxy, p.n_spots, p.snr_hi
        FROM i LEFT JOIN proxy p ON p.hour = i.hour AND p.cell_lat = floor(i.lat / 5)::INT * 5 AND p.cell_lon = floor(i.lon / 5)::INT * 5
        ORDER BY station_id, hour""").df()
    st["time"] = st["hour"] + pd.Timedelta(minutes=30)
    out = np.full(len(st), np.nan)
    for day, idx in st.groupby(st.time.dt.floor("D")).indices.items():
        r = iri_points_cached(st.time.to_numpy()[idx], st.lat.to_numpy()[idx], st.lon.to_numpy()[idx], f107_trailing(con, str(day)))
        out[idx] = r["mufd"]
    st["iri_mufd"] = out
    st["a_obs"] = st.mufd - st.iri_mufd
    has = st.muf_proxy.notna()
    print(f"# {len(st):,} station-hours in {start}..{end}, {st.station_id.nunique()} stations; proxy available for {has.mean():.1%} of them")
    d = st[has].copy()
    print("\n## 1. Proxy vs ionosonde MUF(3000), same cell & hour")
    print(f"   corr(proxy, MUF) = {np.corrcoef(d.muf_proxy, d.mufd)[0,1]:.3f};  corr(IRI MUF, MUF) = {np.corrcoef(d.iri_mufd, d.mufd)[0,1]:.3f}")
    print("   MUF by proxy band:")
    print(d.groupby("muf_proxy").agg(n=("mufd", "size"), muf_mean=("mufd", "mean"), muf_p10=("mufd", lambda x: x.quantile(.1)), iri_mean=("iri_mufd", "mean")).round(1).to_string())
    print("\n## 2. Anomaly: corr(proxy - IRI MUF, MUF - IRI MUF), by spot activity in the cell-hour")
    d["a_proxy"] = d.muf_proxy - d.iri_mufd
    for lo, hi in ((1, 10), (10, 100), (100, 1e9)):
        e = d[(d.n_spots >= lo) & (d.n_spots < hi)]
        if len(e) > 100:
            print(f"   n_spots in [{lo},{hi}): n={len(e):,}  corr={np.corrcoef(e.a_proxy, e.a_obs)[0,1]:.3f}")
    print("\n## 3. Regression: MUF anomaly ~ proxy anomaly + log spots, R^2")
    X = np.c_[np.ones(len(d)), d.a_proxy, np.log1p(d.n_spots)]
    y = d.a_obs.to_numpy(); beta, *_ = np.linalg.lstsq(X, y, rcond=None); pred = X @ beta
    print(f"   R^2 = {1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum():.3f}   (coef on proxy anomaly {beta[1]:+.3f})")
    print("\n## 4. Forecast value: R^2 of station MUF anomaly at T+h from own anomaly at T, proxy anomaly at T, both")
    s2 = st.set_index(["station_id", "hour"]).sort_index()
    for h in (1, 3, 6, 12, 24):
        fut = s2[["a_obs"]].rename(columns={"a_obs": "y"}).copy()
        fut.index = fut.index.set_levels(fut.index.levels[1] - pd.Timedelta(hours=h), level=1)
        j = s2[["a_obs", "muf_proxy", "iri_mufd", "n_spots"]].join(fut, how="inner").dropna()
        j["a_p"] = j.muf_proxy - j.iri_mufd
        y = j.y.to_numpy(); one = np.ones(len(j))
        def r2(X):
            b, *_ = np.linalg.lstsq(X, y, rcond=None); return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"   h={h:2d}: n={len(j):,}  own {r2(np.c_[one, j.a_obs]):.3f}  proxy {r2(np.c_[one, j.a_p]):.3f}  both {r2(np.c_[one, j.a_obs, j.a_p]):.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
