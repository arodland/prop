import os
import urllib.request, json
import pandas as pd
import numpy as np
import george
from kernel import kernel
from cs import cs_to_stdev, stdev_to_cs
import scipy.optimize as op
from multiprocessing import Pool

from pandas.io.json import json_normalize

# Dourbes, Boulder, Hermanus, Roquetes, Wallops, Grahamstown
stations = [12, 11, 26, 3, 61, 25]
points_per_station = 10000

def get_data(url=os.getenv("HISTORY_URI")):
    with urllib.request.urlopen(url) as res:
        data = json.loads(res.read().decode())

    return data

def make_dataset(station):
    url = 'http://localhost:5502/mixscale.json?station=%d&points=%d' % (station, points_per_station)
    data = get_data(url)
    s = data[0]

    x, y_fof2, y_mufd, y_hmf2, sigma, cslist = [], [], [], [], [], []
    last_tm = None

    for pt in s['history']:
        tm = (pd.to_datetime(pt[0]) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
        cs = pt[1]

        if cs < 10 and cs != -1:
            continue

        sd = cs_to_stdev(cs, adj100=True)
        fof2, mufd, hmf2 = pt[2:5]

        x.append(tm / 86400.)
        y_fof2.append(np.log(fof2))
        y_mufd.append(np.log(mufd))
        y_hmf2.append(np.log(hmf2))
        sigma.append(sd)
        cslist.append(cs)
        last_tm = tm

    ds = {}
    ds['station_id'] = s['id']
    ds['x'] = np.array(x)
    ds['last_tm'] = last_tm
    ds['y_fof2'] = np.array(y_fof2)
    ds['mean_fof2'] = np.mean(ds['y_fof2'])
    ds['y_fof2'] -= ds['mean_fof2']
    ds['y_mufd'] = np.array(y_mufd)
    ds['mean_mufd'] = np.mean(ds['y_mufd'])
    ds['y_mufd'] -= ds['mean_mufd']
    ds['y_hmf2'] = np.array(y_hmf2)
    ds['mean_hmf2'] = np.mean(ds['y_hmf2'])
    ds['y_hmf2'] -= ds['mean_hmf2']
    ds['sigma'] = np.array(sigma)
    ds['cs'] = np.array(cslist)

    ds['gp'] = george.GP(kernel)
    ds['gp'].compute(ds['x'], ds['sigma'] + 1e-3)

    return ds

iter, fev, grev, best = 0, 0, 0, 999
datasets = []


def loss(x):
    (i, p) = x
    ds = datasets[i]
    ret = 0
    ds['gp'].set_parameter_vector(p)
    for metric in ('y_fof2', 'y_mufd', 'y_hmf2'):
        ll = ds['gp'].log_likelihood(ds[metric], quiet=True)
        ll = -ll if np.isfinite(ll) else 1e25
        ret += ll

    return ret

def nll(p):
    global fev
    global best
    fev = fev + 1

    losses = [l for l in pool.imap_unordered(loss, [(i, p) for i in range(len(datasets))] )]
    lsum = sum(losses)

    if lsum < best:
        best = lsum
    return lsum

def grad(x):
    (i, p) = x
    ds = datasets[i]
    ret = 0
    ds['gp'].set_parameter_vector(p)
    for metric in ('y_fof2', 'y_mufd', 'y_hmf2'):
        ret -= ds['gp'].grad_log_likelihood(ds[metric], quiet=True)

    return ret

def grad_nll(p):
    global grev
    grev = grev + 1

    gs = 0
    for g in pool.imap_unordered(grad, [(i, p) for i in range(len(datasets))] ):
        gs += g

    return gs

def cb(p):
    global iter
    iter = iter + 1
    print("# iter=", iter, "fev=", fev, "grev=", grev, "best=", best, "p=", p)

if __name__=='__main__':
    # One pool to fetch the data...
    with Pool(processes=6) as p:
        datasets = [ d for d in p.imap(make_dataset, stations) ]

    # And a new one forked afterwards which has the data as globals
    pool = Pool(processes=6)

    p0 = datasets[0]['gp'].get_parameter_vector()
    print("# Init: ", p0)

    opt_result = op.minimize(nll, p0, jac=grad_nll, method='L-BFGS-B', callback=cb, options={'maxiter': 100})
    print("# RESULT ", opt_result.x)

    for ds in datasets:
        with open("/out/plot-%d.txt" % (ds['station_id']), 'w') as f:
            tm = np.around(ds['x'] * 86400)
            fof2 = np.exp(ds['y_fof2'] + ds['mean_fof2'])
            hmf2 = np.exp(ds['y_hmf2'] + ds['mean_hmf2'])
            mufd = np.exp(ds['y_mufd'] + ds['mean_mufd'])
            sd = ds['sigma']
            cs = ds['cs']

            for i in range(len(tm)):
                print("%d\t%f\t%f\t%f\t%f\t%d" % (tm[i], fof2[i], mufd[i], hmf2[i], sd[i], cs[i]), file=f)

            print("", file=f)

            ds['gp'].set_parameter_vector(opt_result.x)
            ds['gp'].compute(ds['x'], ds['sigma'] + 1e-3)

            tm = np.array(range(ds['last_tm'] - (86400 * 2), ds['last_tm'] + (86400 * 7) + 1, 300))

            pred_fof2, sd_fof2 = ds['gp'].predict(ds['y_fof2'], tm / 86400., return_var=True)
            pred_fof2 += ds['mean_fof2']
            sd_fof2 = np.sqrt(sd_fof2)
            pred_mufd, sd_mufd = ds['gp'].predict(ds['y_mufd'], tm / 86400., return_var=True)
            pred_mufd += ds['mean_mufd']
            sd_mufd = np.sqrt(sd_mufd)
            pred_hmf2, sd_hmf2 = ds['gp'].predict(ds['y_hmf2'], tm / 86400., return_var=True)
            pred_hmf2 += ds['mean_hmf2']
            sd_hmf2 = np.sqrt(sd_hmf2)

            for i in range(len(tm)):
                print("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d" % (
                    tm[i],
                    np.exp(pred_fof2[i]), np.exp(pred_fof2[i]-sd_fof2[i]), np.exp(pred_fof2[i]+sd_fof2[i]),
                    np.exp(pred_mufd[i]), np.exp(pred_mufd[i]-sd_mufd[i]), np.exp(pred_mufd[i]+sd_mufd[i]),
                    np.exp(pred_hmf2[i]), np.exp(pred_hmf2[i]-sd_hmf2[i]), np.exp(pred_hmf2[i]+sd_hmf2[i]),
                    sd_mufd[i], stdev_to_cs(sd_mufd[i])
                ), file=f)

    print("# learned parameters = ", opt_result.x)
    kernel.set_parameter_vector(opt_result.x)
    print("# learned kernel = ", kernel)
