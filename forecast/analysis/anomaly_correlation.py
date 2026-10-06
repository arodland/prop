"""Empirical space-time structure of the foF2 anomaly (obs - IRI), to compare with a model's impulse response
(analysis/probe_impulse.py): how fast a station's anomaly decorrelates with lag, how far it reaches, and whether
it moves with local time (westward 15 deg/h) or stays earth-fixed.

    uv run analysis/anomaly_correlation.py /kass/forecast/eval/mids_ladder/y2009.parquet ... [--start 2009 --end 2019]

Input: replay files (eval/replay.py), whose IRI rows give obs and IRI for every sounding. Steps:
 1. hourly mean anomaly per station cluster (UTC hour);
 2. remove each cluster's mean per (year-month, local-time hour), so persistent biases and their diurnal shape
    don't dominate: what is left are fluctuations, the analogue of the probe's delta;
 3. standardise per cluster and correlate a_A(T) with a_B(T+h) over all hours both have.
Correlations are total (marginal) effects: they include whatever the rest of the network shares with A and B, so
compare their *shape* (decay time, reach, drift) with the probe, not their size.
"""
import argparse

import duckdb
import numpy as np
import pandas as pd

LAGS = (0, 1, 2, 3, 6, 12, 18, 24)
DIST = np.array([0, 250, 500, 1000, 1500, 2000, 3000, 5000, 20000])
MIN_N = 500  # joint hours for a pair-lag to count


def gc_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    return 6371 * np.arccos(np.clip(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2), -1, 1))


def hourly(paths):
    con = duckdb.connect(); con.execute("SET memory_limit='24GB'; SET threads=8")
    files = ", ".join(f"'{p}'" for p in paths)
    return con.sql(f"""
        WITH s AS (SELECT cluster, time, any_value(lat) AS lat, any_value(lon) AS lon, any_value(truth) - avg(pred) AS anom
                   FROM read_parquet([{files}]) WHERE mode = 'full' AND model = 'iri' AND kind = 'iono' AND var = 'fof2'
                   GROUP BY cluster, time)
        SELECT cluster::BIGINT AS cluster, date_trunc('hour', time) AS hour, avg(lat) AS lat, avg(lon) AS lon, avg(anom) AS anom
        FROM s GROUP BY 1, 2""").df()


def main(a):
    df = hourly(a.paths)
    df = df[(df.hour >= f"{a.start}-01-01") & (df.hour < f"{a.end}-01-01")]
    pos = df.groupby("cluster")[["lat", "lon"]].median()
    df["lt"] = ((df.hour.dt.hour + df.lon / 15.0) % 24).astype(int); df["ym"] = df.hour.dt.to_period("M")
    df["a"] = df.anom - df.groupby(["cluster", "ym", "lt"]).anom.transform("mean")
    X = df.pivot_table(index="hour", columns="cluster", values="a")
    X = X.reindex(pd.date_range(X.index.min(), X.index.max(), freq="h"))
    keep = X.notna().sum() >= 2000; X = X.loc[:, keep]; P = pos.loc[X.columns]
    X = (X - X.mean()) / X.std()
    print(f"{a.start}..{a.end - 1}: {X.shape[1]} clusters with >= 2000 hours, {int(X.notna().sum().sum()):,} station-hours\n")
    lat, lon = P.lat.to_numpy(), ((P.lon.to_numpy() + 180) % 360) - 180
    D = gc_km(lat[:, None], lon[:, None], lat[None], lon[None])
    dlon = ((lon[None] - lon[:, None] + 180) % 360) - 180  # B east of A positive, [A, B]
    dlat = lat[None] - lat[:, None]
    V = X.to_numpy(); M = np.isfinite(V).astype(np.float32); Z = np.nan_to_num(V).astype(np.float32)
    C, N = {}, {}
    for h in LAGS:
        A_, B_ = (Z[:-h], Z[h:]) if h else (Z, Z); MA, MB = (M[:-h], M[h:]) if h else (M, M)
        C[h] = (A_.T @ B_) / np.maximum(MA.T @ MB, 1); N[h] = MA.T @ MB  # C[h][i, j] = corr(a_i(T), a_j(T+h))
    eye = np.eye(len(lat), dtype=bool)

    print("=== same station: corr(a(T), a(T+h))")
    print("  ".join(f"{h:>2}h {np.nanmean(np.where(N[h][eye] >= MIN_N, C[h][eye], np.nan)):.3f}" for h in LAGS if h))

    print("\n=== other stations: mean corr(a_A(T), a_B(T+h)) by distance (km); rows = lag")
    print("lag  " + " ".join(f"{x:>5}-{y:<5}" for x, y in zip(DIST[:-1], DIST[1:])))
    b = np.digitize(D, DIST) - 1
    for h in LAGS:
        ok = (~eye) & (N[h] >= MIN_N)
        print(f"{h:>3}  " + " ".join(f"{np.nanmean(C[h][ok & (b == k)]) if (ok & (b == k)).any() else np.nan:>11.3f}" for k in range(len(DIST) - 1)))

    print("\n=== zonal pairs (|dlat| < 10, both |lat| < 60): corr by signed lon offset of B from A (east +), rows = lag")
    print("     a local-time-tied anomaly seen at A at T is over B at T+h if B is 15h deg *west* of A")
    edges = np.arange(-105, 106, 15); centers = (edges[:-1] + edges[1:]) / 2
    zon = (np.abs(dlat) < 10) & (np.abs(lat)[:, None] < 60) & (np.abs(lat)[None] < 60) & ~eye
    zb = np.digitize(dlon, edges) - 1
    print("lag  " + " ".join(f"{c:>6.0f}" for c in centers) + "   peak  west-east asym (|dlon| 15-60)")
    for h in LAGS:
        ok = zon & (N[h] >= MIN_N)
        row = [np.nanmean(C[h][ok & (zb == k)]) if (ok & (zb == k)).any() else np.nan for k in range(len(centers))]
        w = ok & (dlon < -15) & (dlon >= -60); e = ok & (dlon > 15) & (dlon <= 60)
        asym = np.nanmean(C[h][w]) - np.nanmean(C[h][e])
        print(f"{h:>3}  " + " ".join(f"{x:>6.3f}" for x in row) + f"   {centers[int(np.nanargmax(row))]:>5.0f}  {asym:+.3f}")
    print("\n(n pairs in zonal set:", int(zon.sum()), ")")

    # Partial effect, the like-for-like comparison with the probe (which moves one station and holds the rest):
    # for each target B, ridge-regress a_B(T+h) on every *other* station's a(T) (B held out, as at a map pixel;
    # missing = 0 = the standardised mean) and bin A's coefficient by distance. Rows with B missing are dropped.
    print("\n=== partial effect: ridge coefficient of a_A(T) in predicting a_B(T+h), B held out, by distance; rows = lag")
    print("lag  " + " ".join(f"{x:>5}-{y:<5}" for x, y in zip(DIST[:-1], DIST[1:])) + "   R2(median)")
    for h in (0, 1, 3, 6, 12, 24):
        coef = np.full(D.shape, np.nan); r2 = []
        Xa = Z[:-h] if h else Z
        for j in range(len(lat)):
            y = V[h:, j] if h else V[:, j]; ok = np.isfinite(y)
            if ok.sum() < 5000:
                continue
            others = np.arange(len(lat)) != j
            A_ = Xa[ok][:, others]; y_ = y[ok]
            G = A_.T @ A_ + a.ridge * len(y_) * np.eye(A_.shape[1]); w = np.linalg.solve(G, A_.T @ y_)
            coef[others, j] = w; r2.append(1 - np.mean((y_ - A_ @ w) ** 2) / np.var(y_))
        ok = np.isfinite(coef)
        print(f"{h:>3}  " + " ".join(f"{np.nanmean(coef[ok & (b == k)]) if (ok & (b == k)).any() else np.nan:>11.3f}" for k in range(len(DIST) - 1)) + f"   {np.median(r2):.3f}")

    if a.temporal_out:
        # Hourly temporal kernel at the station itself, for lags 0..24 h: plain autocorrelation, and the station's own
        # ridge coefficient in predicting a_B(T+h) from every station's a(T) *including B* (the rest of the network
        # held fixed: the analogue of the probe's gain at the perturbed station's own location).
        rows = []
        for h in range(25):
            Xa = Z[:-h] if h else Z; MA, MB = (M[:-h], M[h:]) if h else (M, M)
            cc = ((Z[:-h] if h else Z) * (Z[h:] if h else Z)).sum(0) / np.maximum((MA * MB).sum(0), 1)
            for j in range(len(lat)):
                y = V[h:, j] if h else V[:, j]; ok = np.isfinite(y)
                if ok.sum() < 5000:
                    continue
                A_ = Xa[ok]; y_ = y[ok]
                w = np.linalg.solve(A_.T @ A_ + a.ridge * len(y_) * np.eye(A_.shape[1]), A_.T @ y_)
                rows.append(dict(lag=h, cluster=int(X.columns[j]), lat=lat[j], autocorr=float(cc[j]), self_coef=float(w[j])))
        pd.DataFrame(rows).to_csv(a.temporal_out, index=False)
        print(f"\ntemporal kernel ({len(rows)} rows) -> {a.temporal_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("paths", nargs="+"); ap.add_argument("--start", type=int, default=2009); ap.add_argument("--end", type=int, default=2019); ap.add_argument("--ridge", type=float, default=0.01); ap.add_argument("--temporal-out")
    main(ap.parse_args())
