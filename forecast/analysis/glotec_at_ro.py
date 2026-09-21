"""Sample GloTEC at RO observation points (time, lat, lon) -> parquet with the RO truth alongside.

    uv run analysis/glotec_at_ro.py /kass/forecast/glotec /kass/forecast/eval/glotec_ro.parquet
"""
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.load import connect  # noqa: E402


def main(gdir: Path, out: Path):
    con = connect()
    ro = con.execute("SELECT time, lat, lon, fof2, hmf2, source FROM ro WHERE time >= '2025-05-12' ORDER BY time").df()
    ro["day"] = ro.time.dt.strftime("%Y_%m_%d")
    frames = []
    for day, grp in ro.groupby("day"):
        f = gdir / f"GloTEC_TEC_{day}.nc"
        if not f.exists():
            continue
        with nc.Dataset(f) as ds:
            glat, glon = ds["latitude"][:], ds["longitude"][:]
            t = ds["time"][:].astype("int64")
            ti = np.abs(t[None, :] - grp.time.to_numpy().astype("datetime64[s]").astype("int64")[:, None]).argmin(1)
            li = np.abs(glat[None, :] - grp.lat.to_numpy()[:, None]).argmin(1)
            oi = np.abs(((glon[None, :] - grp.lon.to_numpy()[:, None] + 180) % 360) - 180).argmin(1)
            v = {k: np.asarray(ds[k][:])[ti, li, oi] for k in ("TEC", "NmF2", "hmF2", "quality_flag")}
        g = grp[["time", "lat", "lon", "fof2", "hmf2", "source"]].copy()
        g["g_tec"] = v["TEC"]; g["g_fof2"] = np.sqrt(np.clip(v["NmF2"], 0, None) / 1.24e10)
        g["g_hmf2"] = v["hmF2"]; g["qf"] = v["quality_flag"].astype("int8")
        frames.append(g)
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out)
    print(len(df), "rows; qf>0:", round((df.qf > 0).mean(), 3))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
