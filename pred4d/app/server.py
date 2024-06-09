from collections import namedtuple
from flask import Flask, request, make_response
import magnetic
from tinygp import GaussianProcess
from kernel import make_4d_kernel, metric_params
import jax.numpy as jnp
import numpy as np
import jax
import pandas as pd
import json
import urllib.request
import psycopg
import subprocess
import warnings
import io
import os
import h5py
import hdf5plugin
import time
import kernel
import datetime as dt
from scipy.interpolate import RectBivariateSpline

warnings.simplefilter(action='ignore', category=FutureWarning)
jax.config.update('jax_enable_x64', False)

app = Flask(__name__)

iriidx_table = { 'fof2': 3, 'mufd': 5, 'hmf2': 6, 'md': 0 }

def get_history(metric, max_points):
    with urllib.request.urlopen(f"http://localhost:{os.getenv('HISTORY_PORT')}/4d_history.json?metric={metric}&max_points={max_points}") as res:
        data = json.loads(res.read().decode())
    return data

iri = None

def get_iri(metric, lat, lon, ssn, ts):
    global iri

    if iri is None:
        iri = subprocess.Popen(
            ['/build/iri_opt'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )

    tm = dt.datetime.fromtimestamp(float(ts), tz=dt.timezone.utc)

    iri.stdin.write("%f %f %f %s\n" % (lat, lon, ssn, tm.strftime('%Y %m %d %H %M %S')))
    iri.stdin.flush()
    data = [float(x) for x in iri.stdout.readline().split()]
    if metric == 'md':
        return data[5] / data[3]
    else:
        return data[iriidx_table[metric]]

IriMap = namedtuple('IriMap', ['raw', 'spline'])

def get_irimap(metric, run_id, ts):
    url = "http://localhost:{os.getenv('API_PORT')}/irimap.h5?run_id={run_id}&ts={ts}"
    with urllib.request.urlopen(url) as res:
        content = res.read()
        bio = io.BytesIO(content)
        h5 = h5py.File(bio, 'r')
    ds = h5[f"/maps/{metric}"]
    lat = np.linspace(-90, 90, 181)
    lon = np.linspace(-180, 180, 361)
    spline = RectBivariateSpline(lat, lon, ds)
    return IriMap(ds, spline)

def cs_to_stdev(p, cs):
    cs_int, cs_sl, replace_100, replace_m1 = p[:4]
    if cs == 100:
        cs = replace_100
    elif cs == -1:
        cs = replace_m1
    return cs_int * (1. - cs_sl * cs / 100)

def make_dataset(metric, essn, modip_spline, max_points):
    history = get_history(metric, max_points)
    p = kernel.metric_params[metric]
    x = []
    y = []
    sigma = []
    for row in history:
        ts, meas, cs, lat, lon = row
        irival = get_iri(metric, lat, lon, essn, ts)
        x.append(magnetic.gen_coords_pluggable(lat, lon, ts, None, modip_spline.modip))
        y.append(np.log(meas) - np.log(irival))
        sigma.append(cs_to_stdev(p, cs))

    return {
        'x': jnp.array(x),
        'y': jnp.array(y),
        'sigma': jnp.array(sigma),
    }

@app.route("/generate", methods=['POST'])
def generate():
    run_id = request.form.get('run_id')
    tss = request.form.getlist('target')
    if len(tss) == 0:
        tss = [ time.time() ]

    max_points = int(request.form.get('max_points', 10000))
    metric = request.form.get('metric')

    dsn = "dbname='%s' user='%s' host='%s' password='%s'" % (
        os.getenv("DB_NAME"), os.getenv("DB_USER"), os.getenv("DB_HOST"), os.getenv("DB_PASSWORD"))
    con = psycopg.connect(dsn)

    with con.cursor() as cur:
        cur.execute('select ssn from essn where run_id=%s and series=%s order by time desc nulls last limit 1', (run_id, '24h'))
        (essn,) = cur.fetchone()

    modip_spline = magnetic.ModipSpline(dt.datetime.now())

    ds = make_dataset(metric, essn, modip_spline, max_points)
    kern = kernel.make_4d_kernel(kernel.metric_params[metric][4:])
    gp = GaussianProcess(kern, ds['x'], diag=jnp.power(ds['sigma'], 2.))

