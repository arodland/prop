"""Populate the IRI map cache for the issue times a replay will need (both drivers), so replays are fast.

    uv run eval/prebuild_iri.py --start 2024-01-01 --end 2024-04-01 [--step-hours 6]
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.iri_cache import F107_STEP, _iri_maps_level  # noqa: E402
from data.drivers import attach_essn, attach_indices, essn_sfi, f107_trailing  # noqa: E402
from data.load import SNAPSHOT  # noqa: E402

import duckdb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--step-hours", type=float, default=6)
    ap.add_argument("--drivers", default="f107_81,essn", help="which drivers' levels to build (training samples need only f107_81)")
    a = ap.parse_args()
    con = duckdb.connect(); attach_indices(con, SNAPSHOT); attach_essn(con, SNAPSHOT)
    need = set()
    for t in pd.date_range(a.start, a.end, freq=f"{int(a.step_hours * 60)}min", inclusive="left"):
        drivers = a.drivers.split(",")
        for f in ([f107_trailing(con, str(t))] if "f107_81" in drivers else []) + ([essn_sfi(con, str(t))] if "essn" in drivers else []):
            if f is None:
                continue
            lo = int(np.floor(f / F107_STEP)) * F107_STEP
            for day in (t.date(), (t + pd.Timedelta(hours=24)).date()):  # targets span T..T+24h
                need.add((day, lo)); need.add((day, lo + F107_STEP))
    need = sorted(need)
    print(f"{len(need)} (day, level) maps", flush=True)
    for i, (day, lvl) in enumerate(need):
        _iri_maps_level(day, lvl, None or __import__("baselines.iri", fromlist=["HMF2_MODEL"]).HMF2_MODEL)
        if i % 20 == 0:
            print(f"{i}/{len(need)} {day} {lvl}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
