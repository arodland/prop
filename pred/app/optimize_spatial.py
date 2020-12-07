import os
import sys
import urllib.request, json
import pandas as pd
import numpy as np
import george
from george.kernels import Matern32Kernel, Matern52Kernel, ExpSquaredKernel, ConstantKernel
from cs import cs_to_stdev, stdev_to_cs
import scipy.optimize as op
from multiprocessing import Pool

NSAMPLES = 1000

kernel = 0.0809**2 * Matern52Kernel(0.0648, ndim=3) + 0.169**2 * ExpSquaredKernel(0.481, ndim=3)

def get_data(url):
    with urllib.request.urlopen(url) as res:
        data = json.loads(res.read().decode())

    return data

def get_stations():
    data = get_data('http://localhost:%s/stations.json' % (os.getenv('API_PORT')))
    st = {}
    for row in data:
        station = row['station']
        st[station['id']] = { 'latitude': float(station['latitude']), 'longitude': float(station['longitude']) }

    return st

def sph_to_xyz(lat, lon):
    lon = lon * np.pi / 180.
    lat = lat * np.pi / 180.
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    return x, y, z

def get_datasets():
    data = get_data('http://localhost:%s/pred_sample.json?samples=%d' % (os.getenv('API_PORT'), NSAMPLES))
    ds = None
    dss = []

    for row in data:
        if ds is not None and row['run_id'] != ds['run_id']:
            dss.append(ds)
            ds = None

        if ds is None:
            ds = { 'run_id': row['run_id'], 'measurements': [] }

        station = stations.get(row['station_name'])
        if station is None:
            continue

        x, y, z = sph_to_xyz(station['latitude'], station['longitude'])

        meas = {
            'lat': station['latitude'],
            'lon': station['longitude'],
            'x': x,
            'y': y,
            'z': z,
            'fof2': row['fof2'],
            'hmf2': row['hmf2'],
            'mufd': row['mufd'],
            'cs': row['cs'],
            'sigma': cs_to_stdev(row['cs']),
        }

        ds['measurements'].append(meas)

    if ds is not None and len(ds['measurements']) > 0:
        dss.append(ds)

    for ds in dss:
        x = np.array([ meas['x'] for meas in ds['measurements'] ])
        y = np.array([ meas['y'] for meas in ds['measurements'] ])
        z = np.array([ meas['z'] for meas in ds['measurements'] ])
        cs = np.array([ meas['cs'] for meas in ds['measurements'] ])
        sigma = np.array([ meas['sigma'] for meas in ds['measurements'] ])

        fof2 = np.array([ np.log(meas['fof2']) for meas in ds['measurements'] ])
        mean_fof2 = np.mean(fof2)
        fof2 -= mean_fof2

        hmf2 = np.array([ np.log(meas['hmf2']) for meas in ds['measurements'] ])
        mean_hmf2 = np.mean(hmf2)
        hmf2 -= mean_hmf2

        mufd = np.array([ np.log(meas['mufd']) for meas in ds['measurements'] ])
        mean_mufd = np.mean(mufd)
        mufd -= mean_mufd

        ds['arrays'] = {
            'xyz': np.column_stack((x,y,z)),
            'cs': cs,
            'sigma': sigma,
            'fof2': fof2,
            'mean_fof2': mean_fof2,
            'hmf2': hmf2,
            'mean_hmf2': mean_hmf2,
            'mufd': mufd,
            'mean_mufd': mean_mufd,
        }

        ds['gp'] = george.GP(kernel)
        ds['gp'].compute(ds['arrays']['xyz'], ds['arrays']['sigma'] + 1e-3)

    return dss

iter, fev, grev, best = 0, 0, 0, 999
stations = {}
datasets = []


def loss(x):
    (i, p) = x
    ds = datasets[i]
    ret = 0
    ds['gp'].set_parameter_vector(p)
    for metric in ('fof2', 'mufd', 'hmf2'):
        ll = ds['gp'].log_likelihood(ds['arrays'][metric], quiet=True)
        ll = -ll if np.isfinite(ll) else 1e25
        ret += ll / (NSAMPLES * 3.0)

    return ret

def nll(p):
    global fev
    global best
    fev = fev + 1

    losses = [l for l in pool.imap_unordered(loss, [(i, p) for i in range(len(datasets))] )]
#    print("losses:", losses)
    lsum = sum(losses)

    if lsum < best:
        best = lsum
    return lsum

def grad(x):
    (i, p) = x
    ds = datasets[i]
    ret = 0
    ds['gp'].set_parameter_vector(p)
    for metric in ('fof2', 'mufd', 'hmf2'):
        ret -= ds['gp'].grad_log_likelihood(ds['arrays'][metric], quiet=True) / (NSAMPLES * 3.0)

    return ret

def grad_nll(p):
    global grev
    grev = grev + 1

    gs = 0
    for g in pool.imap_unordered(grad, [(i, p) for i in range(len(datasets))] ):
        gs += g

#    print("gs:", gs)
    return gs

def cb(p):
    global iter
    iter = iter + 1
    print("# iter=", iter, "fev=", fev, "grev=", grev, "best=", best, "p=", p)

if __name__=='__main__':
    stations = get_stations()
    datasets = get_datasets()

    # And a new one forked afterwards which has the data as globals
    pool = Pool(processes=6)

    p0 = datasets[0]['gp'].get_parameter_vector()
    print("# Init: ", p0)

    opt_result = op.minimize(nll, p0, jac=grad_nll, method='L-BFGS-B', callback=cb, options={'maxiter': 100})

    print("# learned parameters = ", opt_result.x)
    kernel.set_parameter_vector(opt_result.x)
    print("# learned kernel = ", kernel)
