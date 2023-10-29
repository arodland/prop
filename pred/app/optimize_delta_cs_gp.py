import os
import sys
import urllib.request, json
import pandas as pd
import numpy as np
import george
from kernel import delta_kernel_noconstant
from skopt import optimizer
import scipy.optimize as op
from multiprocessing import Pool
import subprocess
import pprint
import random

from pandas import json_normalize

# Boulder, Eareckson, Juliusruh, Austin, Dourbes
stations = [11]*10 + [68]*10 + [10]*10 + [1]*10 + [12]*10 + [64]*10 + [46]*10 + [18]*10 + [58]*10 + [66]*10
points_per_station = 5000
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
        print("names:", ds['gp'].get_parameter_names())

        return ds
    except Exception as e:
        print(e)
        return {}

iter, fev, best, besti = 0, 0, float('inf'), 0
datasets = []


def loss(x):
    (i, p) = x
    cs_int, cs_sl = p[:2]
    def cs_to_stdev(cs):
        if cs == -1:
            cs = 75
        elif cs == 100:
            cs = 75
        return cs_int * (1. - cs_sl*cs/100)

    ds = datasets[i]
    ret = 0
    ds['gp'].set_parameter_vector(p[2:])
    sigma = np.vectorize(cs_to_stdev)(ds['cs'])
    ds['gp'].compute(ds['x'], sigma + 1e-3)

    ll = ds['gp'].log_likelihood(ds['y'], quiet=True)
    ll = -ll if np.isfinite(ll) else 1e25
    return ll

def nll(p):
    global fev
    global best
    global besti, iter
    fev = fev + 1

    print("nll:", p, end="")
    perm = sum([random.sample(range(10*x, 10*x+10), k=4) for x in range(len(stations)//10)], [])

    losses = [l for l in pool.imap_unordered(loss, [(i, p) for i in perm] )]
    lsum = sum(losses)

    if lsum < best:
        best = lsum
        besti = iter

    print(" =", lsum)
    return lsum

def cb(res):
    global iter
    iter = iter + 1
    print("# iter=", iter, "fev=", fev, "best=", best, "besti=", besti, "p=", res.x)

if __name__=='__main__':
    # One pool to fetch the data...
    with Pool(processes=10) as p:
        datasets = [ d for d in p.imap(make_dataset, stations) ]

    # And a new one forked afterwards which has the data as globals
    pool = Pool(processes=10)


#    p0 = datasets[0]['gp'].get_parameter_vector()
#    p0 = [0.185, 0.00173, *p0]
#    p0 = [ 1.24532087e-01, 1.07600199e-03, -4.66390199e+00, 6.41513982e+00, 2.96960118e+01, -6.22884435e+00,  2.09614113e+00, -4.75045184e+00, -7.00132378e-01, -6.31072698e+00, -6.42533237e+00]
#    p0 = [ 9.23931937e-02, 8.11855245e-04, -4.54765687e+00, 6.70213568e+00, 3.01382255e+01, -5.98675126e+00,  2.11147572e+00, -4.18921780e+00, -8.58388133e-01, -7.44350343e+00, -6.42626519e+00]
#    p0 = [0.2114932590009682, 0.00215200398, -8.68983787273589, 6.781234943804359, 21.48317087461395, -9.341889847830897, 2.0659380650256542, -4.219604234791644, -1.0967079006697877, -6.592238448885249, -10.733718437704612]
#    p0 = [0.115491492125, 0.0008516045739476519, -5.6845710875, 8.299903305418384, 24.110580400000003, -4.7894010080000005, 1.689180576, -4.691988521426866, -1.07298516625, -5.954802744, -8.0328314875]
#    p0 = [0.13364973029670596, 0.001064505717434565, -6.038910782496048, 7.326526948167502, 20.25440304472476, -5.9481373683971475, 1.6822417837846197, -4.450970258838374, -1.3412314578124802, -5.250370035429094, -9.65341601947409]
#    p0 = [0.15496299923876028, 0.0013306321467932063, -6.0247893847125455, 7.350403720135579, 16.20352243577981, -5.575866874505304, 1.7332640969608568, -4.294692847773042, -1.6765393222656002, -4.96717694721003, -7.725957832049861]
#    p0 = [0.17062585780808812, 0.0014353730992290319, -7.530986730890682, 9.188004650169473, 20.25440304472476, -4.460693499604243, 2.166580121201071, -4.342465068056303, -1.3412314578124802, -4.8625125002258045, -7.176593866247516]
#    p0 = [0.22444821028582246, 0.0021921588093204904, -9.30550621066427, 4.594002325084737, 32.14977967438212, -3.1582278544055784, 2.8315556961840054, -3.564092678029956, -1.132615196496985, -3.576934735763432, -6.782233235856058]
#    p0 = [0.23601040267578316, 0.002328561129720239, -10.789459441488543, 3.7953409352983742, 26.501420818243158, -1.5791139272027892, 5.663111392368011, -3.1213022625770694, -1.4201200796827131, -3.5336641895952003, -3.391116617928029] ## fof2
#    p0 = [0.2677683312184253, 0.0023852488665155576, -8.871472154869895, 5.399004204800802, 29.033339079925362, -2.9589831113088687, 6.589234681916443, -5.997594405612436, -0.9453992799952609, -3.3311521540179796, -4.124041023133191] ## hmf2
#    p0 = [0.17456658423661595, 0.0015994047158731145, -16.3716309140207, 6.860209515776329, 51.18475914552426, -2.6649753904811257, 4.81947025522403, -4.830226728453731, -1.3638295994416558, -6.663251469535183, -5.0312312888292325] ## mufd

#    p0 = [0.21879227576759736, 0.9760550051725823, -3.4820130298066205, 8.872594997632639, 8.21187292140912, -4.718420625503165, -5.22976606371932, -4.4832800915712365, -2.193977998435992, -3.8369632594931016, 0.4364835551458972] ## fof2 2023-08-21
    p0 = [0.22235171520011915, 0.9536767599362671, -2.3620048847379596, 9.07017810957569, 23.304315765733804, -4.882702814375971, 2.315426047847085, -3.9695296234533712, -0.9329864323611465, -5.862572597852539] ## mufd 2023-08-21
#    p0 = [0.21315160999708233, 0.9920850108292543, -4.114045483732827, 8.952038839486693, 24.924526513867004, -4.656695828046397, 6.843946847535253, -3.8215812019159054, -2.9503694403241747, -5.160553945377263, -1.5182670426604803] ## fof2 2023-08-22
#    p0 = [0.24213093225134258, 0.9459521822390862, -4.378704931127611, 7.879567955068222, 46.09140259947029, -4.00304960419496, 9.169721581091721, -4.996060591327506, -1.7191316370807712, -7.382865917914167, -2.051397275167375] ## hmf2 2023-08-22

    p0 = [0.2881264866664164, 0.9833790253121099, -4.445812596399985, 8.690076453451091, 13.469412707441636, -4.490600820529074, -2.3368334889814646, -4.575641358901502, 0.5198806787012304, -8.284487864218535]

    bounds = [[0.15, 1.], [1e-5, 1.]]
    for n in datasets[0]['gp'].get_parameter_names():
        if n.endswith(':log_constant'):
            bounds.append([-5., 1.])
        elif n.endswith(':log_M_0_0'):
            bounds.append([-10., 10.])
        elif n.endswith(':gamma'):
            bounds.append([1e-3, 50.])
        elif n.endswith(':log_alpha'):
            bounds.append([-3., 3.])
        else:
            raise("unknown param")

#    bounds = [sorted([x*0.5, x*2]) for x in p0]

#    print("# Init: ", p0)

    opt_result = optimizer.gp_minimize(nll, bounds, x0=p0, callback=cb, n_initial_points=44, initial_point_generator='lhs', n_calls=500, n_jobs=16, model_queue_size=2, acq_optimizer='sampling')
    print("# RESULT ", opt_result.x)

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
            return cs_int * (1. - cs_sl*cs/100)

        def stdev_to_cs(sd):
            cs = 100 * (cs_int - sd) / cs_sl
            if cs < 0:
                cs = 0
            if cs > 100:
                cs = 100

            return round(cs)

        for i in range(len(x)):
            print("%g\t%g" % (x[i], y[0][i]), file=f)

    for i, ds in enumerate(datasets):
        with open("/out/plot-%d-%d.txt" % (ds['station_id'], i), 'w') as f:
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

    print("# learned parameters = ", opt_result.x)
    delta_kernel_noconstant.set_parameter_vector(opt_result.x[2:])
    print("# learned kernel = ", delta_kernel_noconstant)
