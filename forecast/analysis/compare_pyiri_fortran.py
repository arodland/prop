"""PyIRI vs the Fortran IRI values production stored at RO points (cosmic_eval fof2_iri/hmf2_iri).

    uv run analysis/compare_pyiri_fortran.py [n_sample]

The Fortran ran with CHAIN's projected indices at forecast time; PyIRI runs with each as-of-T
driver from data/drivers.py. Differences mix engine and driver; skill vs RO truth says which matters.
"""
import sys
from pathlib import Path

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.iri import iri_points  # noqa: E402
from data.drivers import DRIVERS, attach_indices  # noqa: E402

SNAP = Path("/kass/forecast/snapshot/2026-09-02/parquet")
n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
year_filter = f"AND year(time) = {sys.argv[2]}" if len(sys.argv) > 2 else ""

con = duckdb.connect()
attach_indices(con, SNAP)
df = con.execute(f"""
    SELECT time, lat, lon, fof2, hmf2, fof2_iri, hmf2_iri FROM '{SNAP}/ro_iri_ref.parquet'
    WHERE fof2_iri IS NOT NULL AND NOT isnan(fof2) {year_filter} USING SAMPLE {n} ROWS (reservoir, 42) ORDER BY time
""").df()
times = df["time"].to_numpy().astype("datetime64[s]")
days = times.astype("datetime64[D]")

def rmse(a, b): return float(np.sqrt(np.nanmean((a - b) ** 2)))
def bias(a, b): return float(np.nanmean(a - b))

print(f"{len(df)} RO points, {df.time.min().date()} .. {df.time.max().date()}")
print(f"{'model':>22} | fof2: bias  rmse vs fortran | rmse vs truth || hmf2: bias  rmse vs fortran | rmse vs truth")
print(f"{'fortran (CHAIN idx)':>22} |        -     -             | {rmse(df.fof2_iri, df.fof2):5.2f}         ||        -     -             | {rmse(df.hmf2_iri, df.hmf2):5.1f}")
variants = [(f"{d} {h}", DRIVERS[d], h) for d in ("f107_81",) for h in ("SHU2015", "AMTB2013", "BSE1979")]
variants += [(d, DRIVERS[d], "SHU2015") for d in DRIVERS if d != "f107_81"]
for name, drv, hm_model in variants:
    fo = np.full(len(df), np.nan); hm = fo.copy()
    for day in np.unique(days):
        idx = np.nonzero(days == day)[0]
        f107 = drv(con, str(day))
        r = iri_points(times[idx], df.lat.to_numpy()[idx], df.lon.to_numpy()[idx], f107, hmf2_model=hm_model)
        fo[idx], hm[idx] = r["fof2"], r["hmf2"]
    print(f"{'pyiri ' + name:>22} | {bias(fo, df.fof2_iri):+5.2f} {rmse(fo, df.fof2_iri):5.2f}             | {rmse(fo, df.fof2):5.2f}         || {bias(hm, df.hmf2_iri):+5.1f} {rmse(hm, df.hmf2_iri):5.1f}             | {rmse(hm, df.hmf2):5.1f}")
    df[f"fof2_{name}"] = fo; df[f"hmf2_{name}"] = hm
df.to_parquet(Path(__file__).with_suffix(".parquet"))
