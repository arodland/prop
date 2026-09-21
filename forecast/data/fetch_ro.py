"""Mirror UCAR GNSS-RO ionPrf daily tarballs and reduce each to a parquet of F2-peak observations.

    nohup uv run data/fetch_ro.py /kass/forecast/ro > /kass/forecast/ro/fetch.log 2>&1 &

One request every SLEEP seconds, one file per day per source; rerunnable (skips done and known-missing days).
Layout: <out>/raw/<source>/<year>/<doy>.tar.gz (or .missing), <out>/parquet/<source>/<year>-<doy>.parquet
"""
import datetime as dt
import io
import sys
import tarfile
import time
from pathlib import Path

import netCDF4
import pandas as pd
import requests

SOURCES = {
    "cosmic-2": ("https://data.cosmic.ucar.edu/gnss-ro/cosmic2/provisional/spaceWeather/level2/{y}/{d:03d}/ionPrf_prov1_{y}_{d:03d}.tar.gz",
                 dt.date(2019, 10, 1)),
    "planetiq": ("https://data.cosmic.ucar.edu/gnss-ro/planetiq/noaa/nrt/level2/{y}/{d:03d}/ionPrf_nrt_{y}_{d:03d}.tar.gz",
                 dt.date(2023, 4, 5)),
}
SLEEP = 3.0
GPS_EPOCH = dt.datetime(1980, 1, 6)
FIELDS = ("edmaxtime", "edmaxlat", "edmaxlon", "critfreq", "edmaxalt")


def parse(tgz: Path, source: str) -> pd.DataFrame:
    rows = []
    with tarfile.open(tgz, "r:gz") as tf:
        for memb in tf:
            f = tf.extractfile(memb)
            if f is None:
                continue
            try:
                ds = netCDF4.Dataset("m.nc", memory=f.read())
                vals = [float(getattr(ds, k)) for k in FIELDS]
            except Exception as e:  # a bad profile is not a bad day
                print(f"  skip {memb.name}: {e}", flush=True)
                continue
            rows.append((GPS_EPOCH + dt.timedelta(seconds=vals[0]), vals[1], vals[2], vals[3], vals[4], source, memb.name))
    return pd.DataFrame(rows, columns=["time", "lat", "lon", "fof2", "hmf2", "source", "file"])


def main(out: Path):
    sess = requests.Session()
    sess.headers["User-Agent"] = "kc2gprop-forecast-mirror (andrew@cleverdomain.org)"
    today = dt.date.today()
    for source, (url_t, start) in SOURCES.items():
        day = start
        while day < today:
            y, d = day.year, day.timetuple().tm_yday
            raw = out / "raw" / source / str(y) / f"{d:03d}.tar.gz"
            pq = out / "parquet" / source / f"{y}-{d:03d}.parquet"
            if not pq.exists() and not raw.with_suffix(".missing").exists():
                if not raw.exists():
                    raw.parent.mkdir(parents=True, exist_ok=True)
                    r = sess.get(url_t.format(y=y, d=d), timeout=300)
                    time.sleep(SLEEP)
                    if r.status_code == 404:
                        raw.with_suffix(".missing").touch()
                        print(f"{source} {day} missing", flush=True)
                        day += dt.timedelta(days=1)
                        continue
                    r.raise_for_status()
                    raw.with_suffix(".tmp").write_bytes(r.content)
                    raw.with_suffix(".tmp").rename(raw)
                df = parse(raw, source)
                pq.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(pq)
                print(f"{source} {day} {len(df)} profiles", flush=True)
            day += dt.timedelta(days=1)
    print("done", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
