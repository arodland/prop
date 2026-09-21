"""Sample every GloTEC 10-min map at the ionosonde station locations -> one parquet.

    uv run analysis/glotec_at_stations.py /kass/forecast/glotec /kass/forecast/eval/glotec_stations.parquet

Columns: time, station_id, tec, fof2 (from NmF2), hmf2, anomaly (GloTEC's own 30-day-median TEC anomaly), qf.
Nearest 2.5° cell; no interpolation (the quality flag is per cell).
"""
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.load import connect  # noqa: E402


def main(gdir: Path, out: Path):
    st = connect().execute("SELECT id, lat, lon FROM station").df()
    frames = []
    for f in sorted(gdir.glob("GloTEC_TEC_*.nc")):
        with nc.Dataset(f) as ds:
            glat, glon = ds["latitude"][:], ds["longitude"][:]
            li = np.abs(glat[None, :] - st.lat.to_numpy()[:, None]).argmin(1)
            oi = np.abs(((glon[None, :] - st.lon.to_numpy()[:, None] + 180) % 360) - 180).argmin(1)
            t = pd.to_datetime(ds["time"][:].astype("int64"), unit="s")
            # netCDF4 fancy indexing is orthogonal (outer product); index numpy arrays for pairwise picks
            v = {k: np.asarray(ds[k][:])[:, li, oi] for k in ("TEC", "NmF2", "hmF2", "anomaly", "quality_flag")}
            frames.append(pd.DataFrame({
                "time": np.repeat(t, len(st)), "station_id": np.tile(st.id.to_numpy(), len(t)),
                "tec": v["TEC"].ravel(),
                "fof2": np.sqrt(np.clip(v["NmF2"], 0, None) / 1.24e10).ravel(),  # MHz from m^-3
                "hmf2": v["hmF2"].ravel(),
                "anomaly": v["anomaly"].ravel(),
                "qf": v["quality_flag"].ravel().astype("int8"),
            }))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out)
    print(len(df), "rows", df.time.min(), df.time.max(), "qf>0:", round((df.qf > 0).mean(), 3))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
