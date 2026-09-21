"""Phase 3 source-value check for GloTEC (PLAN.md): does it carry information about the anomaly (obs - IRI)?

    uv run analysis/glotec_value.py

Three questions, all on 2025-05-12 -> 2026-09-02:
 1. Nowcast at stations: RMSE vs ionosonde foF2 for IRI, GloTEC, by quality flag; corr of anomalies.
 2. Nowcast away from stations: same at RO points (GloTEC vs RO truth vs IRI).
 3. Forecast value: does GloTEC anomaly at T predict the station anomaly at T+h beyond the station's
    own anomaly at T? Partial correlation / incremental R^2 by lead h.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.iri_cache import iri_points_cached  # noqa: E402
from data.drivers import attach_indices, f107_trailing  # noqa: E402
from data.load import SNAPSHOT, connect  # noqa: E402

START = "2025-05-12"


def add_iri(con, df, tcol="time"):
    """IRI foF2/hmF2 at each row, with the trailing-81-day driver of the row's day (as-of-T honest)."""
    out = np.full((len(df), 2), np.nan)
    days = df[tcol].dt.floor("D")
    for day, idx in df.groupby(days).indices.items():
        f107 = f107_trailing(con, str(day))
        r = iri_points_cached(df[tcol].to_numpy()[idx], df.lat.to_numpy()[idx], df.lon.to_numpy()[idx], f107)
        out[idx, 0], out[idx, 1] = r["fof2"], r["hmf2"]
    df["iri_fof2"], df["iri_hmf2"] = out[:, 0], out[:, 1]
    return df


def rmse(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main():
    con = connect(); con.execute("SET memory_limit='24GB'; SET TimeZone='UTC'")
    attach_indices(con, SNAPSHOT)
    con.execute("CREATE VIEW g AS SELECT *, floor((epoch(time::TIMESTAMP) - 300) / 600)::BIGINT AS b FROM '/kass/forecast/eval/glotec_stations.parquet'")
    # hourly station sample: one ionosonde row per station per hour (nearest to the hour), matched to GloTEC
    st = con.execute(f"""
        WITH i AS (SELECT i.station_id, i.time, i.fof2, i.hmf2, s.lat, s.lon, floor((epoch(i.time) - 300) / 600)::BIGINT AS b,
                          row_number() OVER (PARTITION BY i.station_id, date_trunc('hour', i.time) ORDER BY abs(extract(minute FROM i.time) - 0)) AS rn
                   FROM iono i JOIN station s ON s.id = i.station_id WHERE i.time >= '{START}' AND i.fof2 IS NOT NULL)
        SELECT i.station_id, i.time, i.fof2, i.hmf2, i.lat, i.lon, g.fof2 AS g_fof2, g.hmf2 AS g_hmf2, g.qf
        FROM i JOIN g USING (station_id, b) WHERE rn = 1 ORDER BY station_id, time""").df()
    st = add_iri(con, st)
    st["a_obs"] = st.fof2 - st.iri_fof2
    st["a_g"] = st.g_fof2 - st.iri_fof2
    print(f"# Stations: {len(st):,} hourly rows, {st.station_id.nunique()} stations, {START}..{st.time.max().date()}")
    print("\n## 1. Nowcast foF2 RMSE at stations by GloTEC quality flag")
    rows = []
    for qf, d in st.groupby("qf"):
        rows.append({"qf": qf, "n": len(d), "iri": rmse(d.iri_fof2, d.fof2), "glotec": rmse(d.g_fof2, d.fof2),
                     "anom_corr": corr(d.a_obs, d.a_g), "glotec_hmf2": rmse(d.g_hmf2, d.hmf2), "iri_hmf2": rmse(d.iri_hmf2, d.hmf2)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n## 2. Nowcast foF2 RMSE at RO points (away from stations) by quality flag")
    ro = pd.read_parquet("/kass/forecast/eval/glotec_ro.parquet")
    ro = add_iri(con, ro)
    rows = []
    for qf, d in ro.groupby("qf"):
        rows.append({"qf": qf, "n": len(d), "iri": rmse(d.iri_fof2, d.fof2), "glotec": rmse(d.g_fof2, d.fof2),
                     "anom_corr": corr(d.fof2 - d.iri_fof2, d.g_fof2 - d.iri_fof2),
                     "glotec_hmf2": rmse(d.g_hmf2, d.hmf2), "iri_hmf2": rmse(d.iri_hmf2, d.hmf2)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n## 3. Forecast value at stations: predicting anomaly at T+h (qf>=3 at T)")
    print("   R^2 of a_obs(T+h) from: [a_obs(T)] alone, [a_g(T)] alone, [a_obs(T), a_g(T)] together; 'gain' = incremental R^2 from GloTEC")
    st = st.set_index(["station_id", "time"]).sort_index()
    rows = []
    for h in (1, 3, 6, 12, 24):
        fut = st[["a_obs"]].rename(columns={"a_obs": "y"}).copy()
        fut.index = fut.index.set_levels(fut.index.levels[1] - pd.Timedelta(hours=h), level=1)
        j = st[["a_obs", "a_g", "qf"]].join(fut, how="inner")
        j = j[(j.qf >= 3) & np.isfinite(j.a_g)].dropna()
        X1 = np.c_[np.ones(len(j)), j.a_obs]; X2 = np.c_[np.ones(len(j)), j.a_g]; X3 = np.c_[np.ones(len(j)), j.a_obs, j.a_g]
        y = j.y.to_numpy()
        def r2(X):
            beta, *_ = np.linalg.lstsq(X, y, rcond=None); return 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        a, b, c = r2(X1), r2(X2), r2(X3)
        rows.append({"lead_h": h, "n": len(j), "r2_own": a, "r2_glotec": b, "r2_both": c, "gain": c - a})
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
