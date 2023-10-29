import os
import sys
import urllib.request, json
import pandas as pd
import numpy as np
import george
from kernel import delta_kernel_noconstant
import scipy.optimize as op
from multiprocessing import Pool
import subprocess
import pprint

from pandas import json_normalize

# Boulder, Eareckson, Juliusruh, Austin, Dourbes, Hermanus
stations = [11, 68, 10, 1, 12, 64, 46, 18, 58, 66]
points_per_station = 8000
max_span = 365
metric = sys.argv[1]
iriidx_table = { 'fof2': 3, 'mufd': 5, 'hmf2': 6 }
iriidx = iriidx_table[metric]

def get_data(url=os.getenv("HISTORY_URI")):
    with urllib.request.urlopen(url) as res:
        data = json.loads(res.read().decode())

    return data

def make_dataset(station):
    try:
        url = 'http://localhost:5502/mixscale_metric.json?station=%d&points=%d&max_span=%d&metric=%s' % (station, points_per_station, max_span, metric)
        data = get_data(url)
        s = data[0]

        x, y, sigma, cslist = [], [], [], []
        last_tm = None

        iri = subprocess.Popen(
                ['/build/iri_opt'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                universal_newlines=True,
            )

        for pt in s['history']:
            tm = (pd.to_datetime(pt[0]) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
            cs = pt[1]
            iritime = pd.to_datetime(pt[0]).strftime('%Y %m %d %H %M %S')

            if cs < 40 and cs != -1:
                continue

            meas = pt[2]

            if meas <= 0 or not np.isfinite(meas):
                continue

            iri.stdin.write("%f %f %f %s\n" % (s['latitude'], s['longitude'], -100.0, iritime))
            iri.stdin.flush()
            iridata = [float(x) for x in iri.stdout.readline().split()]

            irival = iridata[iriidx]
            if irival <= 0 or not np.isfinite(irival):
                continue

            x.append(tm / 86400.)
            y.append(np.log(meas) - np.log(irival))
            cslist.append(cs)
            last_tm = tm

        iri.stdin.close()
        iri.wait()

        ds = {}
        ds['station_id'] = s['id']
        ds['latitude'] = s['latitude']
        ds['longitude'] = s['longitude']

        ds['x'] = np.array(x)
        ds['last_tm'] = last_tm
        ds['y'] = np.array(y)
        ds['mean'] = np.mean(ds['y'])
        ds['y'] -= ds['mean']
        ds['cs'] = np.array(cslist)

        ds['gp'] = george.GP(delta_kernel_noconstant)

        return ds
    except Exception as e:
        print(e)
        return {}

iter, fev, grev, best = 0, 0, 0, 999
datasets = []


def loss(x):
    (i, p) = x
    cs_int, cs_sl = p[:2]
    if cs_sl > 1.:
        return 1e24

    def cs_to_stdev(cs):
        if cs == -1:
            cs = 75
        elif cs == 100:
            cs = 75
        sigma = cs_int * (1. - cs_sl*cs/100)
        if sigma < 1e-3:
            return 1e-3
        return sigma

    ds = datasets[i]
    ret = 0
    ds['gp'].set_parameter_vector(p[2:])
    sigma = np.vectorize(cs_to_stdev)(ds['cs'])
    ds['gp'].compute(ds['x'], sigma)

    ll = ds['gp'].log_likelihood(ds['y'], quiet=True)
    ll = -ll if np.isfinite(ll) else 1e25
    return ll

def nll(p):
    global fev
    global best
    fev = fev + 1

    losses = [l for l in pool.imap_unordered(loss, [(i, p) for i in range(len(datasets))] )]
    lsum = sum(losses)

    if lsum < best:
        best = lsum
    return lsum

def cb(p):
    global iter
    iter = iter + 1
    print("# iter=", iter, "fev=", fev, "grev=", grev, "best=", best, "p=", list(p))

if __name__=='__main__':
    # One pool to fetch the data...
    with Pool(processes=5) as p:
        datasets = [ d for d in p.imap(make_dataset, stations) ]

    # And a new one forked afterwards which has the data as globals
    pool = Pool(processes=10)

#    p0 = datasets[0]['gp'].get_parameter_vector()
#    p0 = [0.185, 0.00173, *p0]
#    p0 = [ 1.24532087e-01, 1.07600199e-03, -4.66390199e+00, 6.41513982e+00, 2.96960118e+01, -6.22884435e+00,  2.09614113e+00, -4.75045184e+00, -7.00132378e-01, -6.31072698e+00, -6.42533237e+00]
#    p0 = [0.380843, 1.02325164, -4.20129749, 8.64689724, 25.13420059, -4.73636353, 6.98121323, -3.2742518, -2.80872342, -6.00112368, -1.47149693]
    p0 = [0.3044394984137353, 0.998835375671616, -4.466262813160847, 4.46391161348542, 34.257503381131514, -4.759681080172085, 6.687712857844176, -3.7418986782992536, -2.1605090001229126, -6.12043134858618]

    print("# Init: ", p0)

#    opt_result = op.minimize(nll, p0, method='TNC', callback=cb, options={'maxfun': 200})
    opt_result = op.minimize(nll, p0, jac='2-point', method='TNC', callback=cb, options={'maxfun': 1000})
    print("# RESULT ", list(opt_result.x))

    with open('/out/plot-kernel.txt', 'w') as f:
        delta_kernel_noconstant.set_parameter_vector(opt_result.x[2:])
        x = np.linspace(0, 30, 3000)
        y = delta_kernel_noconstant.get_value(np.atleast_2d(x).T)

        cs_int, cs_sl = opt_result.x[:2]
        def cs_to_stdev(cs):
            if cs == -1:
                cs = 75
            elif cs == 100:
                cs = 75
            sigma = cs_int * (1. - cs_sl*cs/100)
            if sigma < 1e-3:
                return 1e-3
            return sigma

        def stdev_to_cs(sd):
            cs = 100 * (cs_int - sd) / cs_sl
            if cs < 0:
                cs = 0
            if cs > 100:
                cs = 100

            return round(cs)

        for i in range(len(x)):
            print("%g\t%g" % (x[i], y[0][i]), file=f)

    for ds in datasets:
        with open("/out/plot-%d.txt" % (ds['station_id']), 'w') as f:
            tm = np.around(ds['x'] * 86400)
            y = np.exp(ds['y'] + ds['mean'])
            cs = ds['cs']
            sd = np.vectorize(cs_to_stdev)(ds['cs'])

            for i in range(len(tm)):
                print("%d\t%f\t%f\t%d" % (tm[i], y[i], sd[i], cs[i]), file=f)

            print("", file=f)

            ds['gp'].set_parameter_vector(opt_result.x[2:])
            ds['gp'].compute(ds['x'], sd + 1e-3)

            tm = np.array(range(ds['last_tm'] - (86400 * 2), ds['last_tm'] + (86400 * 7) + 1, 300))

            pred, sd = ds['gp'].predict(ds['y'], tm / 86400., return_var=True)
            pred += ds['mean']
            sd = np.sqrt(sd)

            for i in range(len(tm)):
                print("%d\t%f\t%f\t%f\t%f\t%d" % (
                    tm[i],
                    np.exp(pred[i]), np.exp(pred[i]-sd[i]), np.exp(pred[i]+sd[i]),
                    sd[i], stdev_to_cs(sd[i])
                ), file=f)

    delta_kernel_noconstant.set_parameter_vector(opt_result.x[2:])
    print("# learned kernel = ", delta_kernel_noconstant)
