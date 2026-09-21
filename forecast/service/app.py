"""Forecast service (PLAN.md Phase 5): the anomaly model as a drop-in producer of production's hourly map files.

POST /forecast_24h  form: run_id, model (checkpoint dir name under $FORECAST_MODELS), glotec (0/1), holdout (0/1)
  -> 25 rows in `assimilated (time, run_id, dataset)`, one per hour T..T+24h, HDF5 in assimilate's layout:
     /maps/{fof2,hmf2,mufd,md,foe,gyf}  /stdev/{fof2,hmf2,mufd}  /stationdata/{curr,pred}  /essn/{ssn,sfi}  /ts
Inputs come straight from Postgres (measurement, station, holdout, essn, runs); GloTEC from the NOAA daily files
(fetched into $GLOTEC_DIR on demand); IRI from PyIRI via the map cache in $IRI_CACHE. No dependency on essn/pred/irimap.
GET /health
"""
import datetime as dt
import io
import threading
import time
import json
import os
import sys
import urllib.request
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
from decimal import Decimal
import pandas as pd
import psycopg
import torch
from flask import Flask, make_response, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "train"))
import PyIRI  # noqa: E402
import PyIRI.igrf_library as igrf  # noqa: E402
from baselines.iri_cache import foe_points_cached, iri_points_cached  # noqa: E402
from build_samples import ANOM_SCALE, F_QRY, VARS, query_feats, query_state, tokens_at  # noqa: E402
from data.load import BOUNDS  # noqa: E402
from glotec_tokens import GLOTEC, glotec_tokens  # noqa: E402
# live spot aggregates from service/spots.py (prop-spots.timer); the baseline is shipped once beside the checkpoints
SPOTS_DIR = Path(os.environ.get("SPOTS_DIR", "/checkpoints/spots"))
os.environ.setdefault("SPOT_AGG_DIR", str(SPOTS_DIR / "agg")); os.environ.setdefault("SPOT_BASELINE", str(SPOTS_DIR / "spot_baseline_cp.parquet"))
import spot_tokens as spot_mod  # noqa: E402
SPOT_CAP = int(os.environ.get("FORECAST_SPOT_CAP", 6000))  # build-time cap of the training samples (v7p6: 6000)
from model import AnomalyModel  # noqa: E402

app = Flask(__name__)
MODELS = Path(os.environ.get("FORECAST_MODELS", "/checkpoints"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAT = np.arange(-90, 91, 1.0); LON = np.arange(-180, 181, 1.0)
_models = {}


def dsn():
    return "dbname='%s' user='%s' host='%s' port='%s' password='%s'" % (os.getenv("DB_NAME"), os.getenv("DB_USER"), os.getenv("DB_HOST"), os.getenv("PGPORT", "5432"), os.getenv("DB_PASSWORD"))


def load_model(name):
    if name not in _models:
        ck = torch.load(MODELS / name / "best.pt", map_location=DEV)
        m = AnomalyModel(d=ck["args"]["d"], enc_layers=ck["args"]["layers"], query_self_attn=ck["args"].get("query_self_attn", True), glotec=ck["args"].get("glotec", False), spots=ck["args"].get("spots", False), f_spot=ck["args"].get("f_spot", 29), f_qry=ck["args"].get("f_qry", 10)).to(DEV)
        m.load_state_dict(ck["model"]); m.eval(); m.pool = bool(ck["args"].get("pool", False)); _models[name] = m  # pooled ionosonde tokens?
        if m.spots and int(ck["args"].get("spot_res", 60)) != spot_mod.RES[0]:
            raise RuntimeError(f"checkpoint {name} was trained on {ck['args'].get('spot_res', 60)}-min spot bins but SPOT_RES={spot_mod.RES[0]}; set SPOT_RES for the loader and the service")
        if m.spots and ck["args"].get("spot_baseline", "frozen") != spot_mod.baseline_scheme():
            raise RuntimeError(f"checkpoint {name} was trained against a {ck['args'].get('spot_baseline', 'frozen')} spot baseline but "
                               f"{os.environ['SPOT_BASELINE']} is {spot_mod.baseline_scheme()}; the anomaly means different things under the two")
    return _models[name]


def query_df(con, sql, params=None):
    """DataFrame from a psycopg query without pandas' SQLAlchemy warning."""
    cur = con.execute(sql, params)
    df = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    for c in df.columns:  # psycopg returns NUMERIC as Decimal; read_sql used to coerce these
        if df[c].dtype == object and df[c].map(lambda v: isinstance(v, Decimal) or v is None).all() and df[c].notna().any():
            df[c] = df[c].astype(float)
    return df


def db_inputs(con, t0, held_station_ids):
    """Last 24 h of ionosonde rows before t0 - 15 min, quality- and bounds-filtered, with station geometry and cluster."""
    st = query_df(con, "SELECT id, latitude::float AS lat, longitude::float AS lon FROM station")
    # clusters as in data/load.py: co-located codes (<50 km) share the smallest id
    la, lo = np.radians(st.lat.to_numpy()), np.radians(st.lon.to_numpy())
    d = 6371 * np.arccos(np.clip(np.sin(la)[:, None] * np.sin(la)[None] + np.cos(la)[:, None] * np.cos(la)[None] * np.cos(lo[:, None] - lo[None]), -1, 1))
    st["cluster"] = [int(st.id.to_numpy()[(d[i] < 50)].min()) for i in range(len(st))]
    m = query_df(con, """SELECT station_id, time, cs, fof2, hmf2, mufd FROM measurement
                         WHERE time > %(a)s AND time <= %(b)s AND (cs >= 75 OR cs = -1)""",
                 {"a": t0 - dt.timedelta(hours=24), "b": t0 - dt.timedelta(minutes=15)})
    for v, (lo_, hi_) in BOUNDS.items():
        if v in m:
            m[v] = m[v].where((m[v] >= lo_) & (m[v] <= hi_))
    m = m.merge(st.rename(columns={"id": "station_id"}), on="station_id")
    held = set(st.loc[st.id.isin(held_station_ids), "cluster"])
    return m[~m.cluster.isin(held)].reset_index(drop=True), st


def global_token(con, t0, f107):
    daily = query_df(con, "SELECT date, f107_obs, ap3h FROM indices_daily ORDER BY date DESC LIMIT 3") if table_exists(con, "indices_daily") else None
    # Indices: prefer the snapshot's daily table if mirrored into the DB; else fall back to F10.7 only.
    f107d, ap_now, ap_max = f107, 5.0, 5.0
    if daily is not None and len(daily):
        r = daily.iloc[0]; f107d = float(r.f107_obs) if np.isfinite(r.f107_obs) else f107
        aps = np.concatenate([np.asarray(x, float) for x in daily.ap3h]); ap_now = float(np.nanmax(aps[:1])); ap_max = float(np.nanmax(aps[:8]))
    doy = t0.timetuple().tm_yday / 365.25 * 2 * np.pi
    return np.array([f107 / 200.0, f107d / 200.0, np.log1p(ap_now) / 5, np.log1p(ap_max) / 5, np.sin(doy), np.cos(doy)], np.float32)


def table_exists(con, name):
    return con.execute("SELECT to_regclass(%s)", (name,)).fetchone()[0] is not None


def f107_driver(con, t0):
    """Trailing 81-day mean F10.7. Uses indices_daily if present, else the latest essn sfi as a stand-in."""
    if table_exists(con, "indices_daily"):
        v = con.execute("SELECT avg(f107_obs) FROM indices_daily WHERE date < %s AND date >= %s", (t0.date(), t0.date() - dt.timedelta(days=81))).fetchone()[0]
        if v is not None:
            return float(v)
    v = con.execute("SELECT sfi FROM essn WHERE series='24h' ORDER BY time DESC LIMIT 1").fetchone()
    return float(v[0]) if v else 150.0


def glotec_latency_min(t0):
    """Age, in minutes, of the newest GloTEC step available at t0 (from the day's file); None if no file."""
    import netCDF4 as nc
    f = GLOTEC / f"GloTEC_TEC_{t0.date():%Y_%m_%d}.nc"
    if not f.exists():
        return None
    with nc.Dataset(f) as ds:
        newest = float(np.asarray(ds["time"][:]).max())
    return round((t0.replace(tzinfo=dt.timezone.utc).timestamp() - newest) / 60, 1)


def ensure_glotec(t0):
    """Fetch the NOAA daily files for today and yesterday into GLOTEC if missing (today's is refreshed)."""
    GLOTEC.mkdir(parents=True, exist_ok=True)
    for day in (t0.date() - dt.timedelta(days=1), t0.date()):
        f = GLOTEC / f"GloTEC_TEC_{day:%Y_%m_%d}.nc"
        if f.exists() and day != t0.date():
            continue
        try:
            urllib.request.urlretrieve(f"https://services.swpc.noaa.gov/products/glotec/netcdf_2d_urt/GloTEC_TEC_{day:%Y_%m_%d}.nc", f.with_suffix(".tmp"))
            f.with_suffix(".tmp").rename(f)
        except Exception as e:  # noqa: BLE001
            print("glotec fetch failed", day, e, flush=True)


def gyf_map(t0):
    yd = t0.year + (t0.timetuple().tm_yday - 1 + t0.hour / 24) / 365.25
    lon, lat = np.meshgrid(LON, LAT)
    F = igrf.inclination(PyIRI.coeff_dir, yd, lon.ravel(), lat.ravel(), alt=100.0, only_inc=False)[-1]
    return (0.000028 * np.asarray(F)).reshape(181, 361)


def foe_maps(t0, f107):
    """foE on the 1° grid for hours 0..24 from T, from the cached 2° PyIRI E-layer maps (one build per day, then ms)."""
    lon, lat = np.meshgrid(LON, LAT)
    out = np.zeros((25, 181, 361), np.float32)
    for k in range(25):
        tv = np.full(lat.size, np.datetime64(t0.replace(tzinfo=None), "s") + np.timedelta64(k * 3600, "s"))
        out[k] = foe_points_cached(tv, lat.ravel(), lon.ravel(), f107).reshape(181, 361)
    return out


@app.route("/health")
def health():
    return f"OK device={DEV}\n"


@app.route("/forecast_24h", methods=["POST"])
def forecast_24h():
    run_id = int(request.form.get("run_id", -1)); name = request.form.get("model", "v4")
    use_glotec = request.form.get("glotec", "0") in ("1", "true", "True"); use_spots = request.form.get("spots", "0") in ("1", "true", "True"); holdout = request.form.get("holdout", "0") in ("1", "true", "True")
    model = load_model(name)
    with psycopg.connect(dsn()) as con:
        t0 = con.execute("SELECT target_time FROM runs WHERE id=%s", (run_id,)).fetchone()[0]
        held = [r[0] for r in con.execute("SELECT station_id FROM holdout WHERE run_id=%s", (run_id,)).fetchall()] if holdout else []
        inp, stations = db_inputs(con, t0, held)
        f107 = f107_driver(con, t0)
        glob = global_token(con, t0, f107)
        essn = con.execute("SELECT ssn, sfi FROM essn WHERE run_id=%s AND series='24h' ORDER BY time DESC LIMIT 1", (run_id,)).fetchone()
        print(f"run {run_id} T={t0} model={name} glotec={use_glotec} spots={use_spots} inputs={len(inp)} rows/{inp.station_id.nunique()} stations f107={f107:.1f} dev={DEV}", flush=True)
        t0n = np.datetime64(t0.replace(tzinfo=None), "s")
        rng = np.random.default_rng(0); tick = [time.time()]
        def lap(label):
            now = time.time(); print(f"  {label}: {now - tick[0]:.1f}s", flush=True); tick[0] = now
        tok, _, inp2 = tokens_at(None, str(t0n), f107, rng, holdout_frac=0.0, max_tok=4096, inp=inp.assign(time=pd.to_datetime(inp.time).dt.tz_localize(None)), pool=model.pool)
        lap("ionosonde tokens (+IRI cache)")
        tok_g = np.zeros((0, 14), np.float32)
        glotec_lat = None
        if use_glotec and model.glotec:
            ensure_glotec(t0); tok_g = glotec_tokens(t0n, f107, rng); glotec_lat = glotec_latency_min(t0)
            if len(tok_g) == 0:
                print(f"  glotec: no usable step within 3 h of T-30 min (NOAA file late/missing); running without GloTEC", flush=True)
            else:
                print(f"  glotec: {len(tok_g)} tokens, newest step {glotec_lat} min before T (training assumes 30; lag-0 tokens carry their true age)", flush=True)
        tok_s = np.zeros((0, model.f_spot), np.float32); spot_lat = None
        if use_spots and model.spots and any((SPOTS_DIR / "agg").glob("*_live.parquet")):
            spot_mod._con = None; spot_mod.CP[0] = model.f_spot == spot_mod.F_SPOT_CP  # fresh views each run: the loader replaces the files
            tok_s = spot_mod.spot_tokens(t0n, rng, max_spot=SPOT_CAP)
            if len(tok_s) == 0:
                views = {r[0] for r in spot_mod.con().execute("SELECT view_name FROM duckdb_views()").fetchall()}
                print(f"  spots: no tokens; baseline {'present' if 'base' in views else 'MISSING at ' + os.environ['SPOT_BASELINE']}, aggregate views {sorted(views & {'wspr', 'psk', 'wspr_cp', 'psk_cp'})}", flush=True)
            newest = spot_mod.con().execute("SELECT max(hour) FROM wspr").fetchone()[0]
            spot_lat = None if newest is None else (t0n - np.datetime64(newest, "s")).astype("timedelta64[m]").astype(int) - spot_mod.RES[0]  # end of newest complete bin
            print(f"  spots: {len(tok_s)} tokens, newest complete {spot_mod.RES[0]}-min bin ended {spot_lat} min before T (tokens use bins ending <= T-{spot_mod.LAG_MIN if spot_mod.RES[0] < 60 else 60} min)", flush=True); lap("spot tokens")
        T = torch.from_numpy(tok)[None].to(DEV); M = torch.ones(1, len(tok), dtype=torch.bool, device=DEV); G = torch.from_numpy(glob)[None].to(DEV)
        TG = torch.from_numpy(tok_g)[None].to(DEV); GM = torch.ones(1, len(tok_g), dtype=torch.bool, device=DEV)
        TS = torch.from_numpy(tok_s)[None].to(DEV); SM = torch.ones(1, len(tok_s), dtype=torch.bool, device=DEV)
        lon, lat = np.meshgrid(LON, LAT); lat_f, lon_f = lat.ravel(), lon.ravel()
        if model.f_qry > F_QRY:  # nearest-station state features (build_samples --qstate), from the same pooled rows the tokens came from
            s_anom = np.stack(inp2.anom_n.to_list()) if len(inp2) else np.zeros((0, 3)); s_t = inp2["time"].to_numpy().astype("datetime64[s]")
            qs_grid = query_state(inp2.lat.to_numpy(), inp2.lon.to_numpy(), s_t, s_anom, t0n, lat_f, lon_f)
            qs_st = query_state(inp2.lat.to_numpy(), inp2.lon.to_numpy(), s_t, s_anom, t0n, stations[stations.id.isin(inp.station_id.unique())].lat.to_numpy(), stations[stations.id.isin(inp.station_id.unique())].lon.to_numpy())
        if use_glotec and model.glotec:
            lap("glotec tokens")
        gyf = gyf_map(t0); foe = foe_maps(t0, f107); lap("foE + gyrofrequency maps (cache)")
        # station points for /stationdata/pred: the input stations
        st_in = stations[stations.id.isin(inp.station_id.unique())]
        curr = inp.sort_values("time").groupby("station_id").last().reset_index().merge(st_in[["id"]], left_on="station_id", right_on="id")
        # renderer contract (renderer.py filter_data / plot.draw_dots): station.latitude/longitude, time in ms, cs in [0, 1]
        curr_json = (curr.rename(columns={"lat": "station.latitude", "lon": "station.longitude"})
                     .assign(time=lambda d: pd.to_datetime(d.time).to_numpy().astype("datetime64[ms]").astype("int64"), cs=lambda d: np.where(d.cs < 0, 0.85, d.cs / 100.0))  # ms, resolution-safe
                     .assign(**{"station.id": lambda d: d.station_id})  # assimilate's json_normalize schema: prop-cosmic reads station.id
                     [["station.id", "station.latitude", "station.longitude", "time", "cs", "fof2", "hmf2", "mufd"]].to_json(orient="records"))
        with torch.no_grad():
            H, PAD = model.encode(T, M, G, TG, GM, TS, SM)  # encoder once; 25 leads x ~8 grid chunks only run the decoder
            for h in range(25):
                tv = t0n + np.timedelta64(h * 3600, "s")
                times = np.full(lat_f.size, tv)
                iri = iri_points_cached(times, lat_f, lon_f, f107); iri_v = np.c_[iri["fof2"], iri["hmf2"], iri["mufd"]]
                q = query_feats(times, lat_f, lon_f, iri_v, np.full(lat_f.size, 2)); q[:, 5] = h / 24.0
                if model.f_qry > F_QRY:
                    q = np.c_[q, qs_grid].astype(np.float32)
                means, sig = [], []
                for i in range(0, len(q), 8192):
                    mu, lv = model.decode(torch.from_numpy(q[i:i + 8192])[None].to(DEV), H, PAD)
                    means.append(mu[0].cpu().numpy()); sig.append(np.exp(0.5 * lv[0].cpu().numpy()))
                fc = iri_v + np.concatenate(means) * ANOM_SCALE; sd = np.concatenate(sig) * ANOM_SCALE
                fc[:, 0] = np.clip(fc[:, 0], 0.5, 25); fc[:, 1] = np.clip(fc[:, 1], 100, 600); fc[:, 2] = np.clip(fc[:, 2], 1, 80)
                # station predictions for the dots
                stt = np.full(len(st_in), tv); si = iri_points_cached(stt, st_in.lat.to_numpy(), st_in.lon.to_numpy(), f107)
                sq = query_feats(stt, st_in.lat.to_numpy(), st_in.lon.to_numpy(), np.c_[si["fof2"], si["hmf2"], si["mufd"]], np.zeros(len(st_in))); sq[:, 5] = h / 24.0
                if model.f_qry > F_QRY:
                    sq = np.c_[sq, qs_st].astype(np.float32)
                smu, slv = model.decode(torch.from_numpy(sq)[None].to(DEV), H, PAD)
                sp = np.c_[si["fof2"], si["hmf2"], si["mufd"]] + smu[0].cpu().numpy() * ANOM_SCALE
                s_fof2 = np.exp(0.5 * slv[0, :, 0].cpu().numpy()) * ANOM_SCALE[0]
                pred_json = pd.DataFrame({"station.latitude": st_in.lat.to_numpy(), "station.longitude": st_in.lon.to_numpy(), "time": int(tv.astype("int64")) * 1000,
                                          "fof2": sp[:, 0], "hmf2": sp[:, 1], "mufd": sp[:, 2], "stdev_fof2": s_fof2,
                                          "cs": np.clip(1.0 - s_fof2 / (2 * ANOM_SCALE[0]), 0.3, 1.0),  # dot alpha: confident = opaque; never filtered out
                                          "run_id": run_id, "station.id": st_in.id.to_numpy()}).to_json(orient="records")
                bio = io.BytesIO()
                with h5py.File(bio, "w") as h5:
                    maps = {"fof2": fc[:, 0], "hmf2": fc[:, 1], "mufd": fc[:, 2], "md": fc[:, 2] / fc[:, 0], "foe": foe[h].ravel(), "gyf": gyf.ravel()}
                    for k, v in maps.items():
                        h5.create_dataset(f"/maps/{k}", data=np.asarray(v, np.float64).reshape(181, 361), **hdf5plugin.SZ(absolute=0.001))
                    for k, j in (("fof2", 0), ("hmf2", 1), ("mufd", 2)):
                        h5.create_dataset(f"/stdev/{k}", data=sd[:, j].astype(np.float64).reshape(181, 361), **hdf5plugin.SZ(absolute=0.0001))
                    h5.create_dataset("/ts", data=np.array(float(tv.astype("int64"))))
                    h5.create_dataset("/essn/ssn", data=np.array(essn[0] if essn else np.nan, np.float32))
                    h5.create_dataset("/essn/sfi", data=np.array(essn[1] if essn else f107, np.float32))
                    h5.create_dataset("/stationdata/curr", data=curr_json); h5.create_dataset("/stationdata/pred", data=pred_json)
                    h5.attrs["model"] = name; h5.attrs["glotec"] = int(use_glotec and len(tok_g) > 0); h5.attrs["spots"] = int(use_spots and len(tok_s) > 0); h5.attrs["n_tokens"] = len(tok); h5.attrs["f107"] = f107
                    h5.attrs["glotec_latency_min"] = -1.0 if glotec_lat is None else float(glotec_lat)
                    h5.attrs["n_spot_tokens"] = len(tok_s); h5.attrs["spot_latency_min"] = -1.0 if spot_lat is None else float(spot_lat)
                con.execute("INSERT INTO assimilated (time, run_id, dataset) VALUES (%s, %s, %s) ON CONFLICT (run_id, time) DO UPDATE SET dataset = excluded.dataset",
                            (dt.datetime.fromtimestamp(int(tv.astype("int64")), dt.timezone.utc), run_id, bio.getvalue()))
        con.commit(); lap("25 hourly maps: inference + write")
    threading.Thread(target=prewarm, args=(t0 + dt.timedelta(days=1), f107), daemon=True).start()
    return make_response("OK\n")


def prewarm(day_after, f107):
    """Build tomorrow's IRI and foE cache maps in the background so the first run of a new day does not pay ~5 min of PyIRI."""
    try:
        from baselines.iri_cache import F107_STEP, _foe_maps_level, _iri_maps_level, HMF2_MODEL
        lo = int(np.floor(f107 / F107_STEP)) * F107_STEP
        for lvl in (lo, lo + F107_STEP):
            _iri_maps_level(day_after.date(), lvl, HMF2_MODEL); _foe_maps_level(day_after.date(), lvl)
    except Exception as e:  # noqa: BLE001
        print("prewarm failed:", e, flush=True)


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=int(os.environ.get("FORECAST_PORT", 5515)), threads=1)
