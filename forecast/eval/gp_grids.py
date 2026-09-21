"""Export an archived production run's maps into the grid format forecast.py saves, so eval/coherence.py
can score the GP maps as the reference (PLAN.md: bounds calibrated on GP maps).

    uv run eval/gp_grids.py --at 2025-06-15T12:00 --step-hours 3 --out /kass/forecast/maps/2025-06-15_gp
"""
import argparse
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.iri_cache import iri_points_cached  # noqa: E402
from data.drivers import attach_indices, f107_trailing  # noqa: E402
from data.load import SNAPSHOT, connect  # noqa: E402
from eval.gp_maps import run_dir  # noqa: E402

LAT = np.arange(-90, 91, 1.0); LON = np.arange(-180, 181, 1.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--at", required=True); ap.add_argument("--step-hours", type=float, default=3); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    con = connect(); attach_indices(con, SNAPSHOT); T = pd.Timestamp(a.at); f107 = f107_trailing(con, str(T))
    run = con.execute(f"SELECT min(id) FROM '{SNAPSHOT}/runs.parquet' WHERE experiment IS NULL AND target_time = ?", [T]).fetchone()[0]
    d = run_dir(int(run)) / "assimilated"; files = {int(round(float(h5py.File(f)["ts"][()]))): f for f in d.glob("*.h5")}
    lon, lat = np.meshgrid(LON, LAT)
    stations = None
    for lead in np.arange(0, 24.001, a.step_hours):
        ts = int((T + pd.Timedelta(hours=float(lead))).timestamp())
        if ts not in files:
            continue
        with h5py.File(files[ts]) as h:
            fc = np.stack([h[f"/maps/{v}"][:].ravel() for v in ("fof2", "hmf2", "mufd")], 1)
            if stations is None:
                import json
                sd = json.loads(h["/stationdata/pred"][()]); stations = np.array([[s["station.latitude"], s["station.longitude"]] for s in sd], np.float32)
        times = np.full(lat.size, np.datetime64(T + pd.Timedelta(hours=float(lead)), "s"))
        iri = iri_points_cached(times, lat.ravel(), lon.ravel(), f107); iri_v = np.c_[iri["fof2"], iri["hmf2"], iri["mufd"]]
        np.savez_compressed(out / f"grid_lead{int(lead):02d}.npz", fc=fc.astype(np.float32), iri=iri_v.astype(np.float32), sigma=np.zeros(0, np.float32),
                            lead_h=float(lead), issue=np.datetime64(T, "s").astype(np.int64), stations=stations)
    print(f"run {run}: {len(list(out.glob('grid_lead*.npz')))} leads -> {out}")


if __name__ == "__main__":
    main()
