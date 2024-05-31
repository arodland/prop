from collections import namedtuple
import matplotlib.pyplot as plt
import subprocess
from magnetic import gen_coords
from tinygp import GaussianProcess
from kernel import make_4d_kernel
import scipy.optimize as op
import jax.numpy as jnp
import jax
import pandas as pd
import json
import urllib.request
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


jax.config.update('jax_enable_x64', True)

metric = sys.argv[1]
iriidx_table = { 'fof2': 3, 'mufd': 5, 'hmf2': 6 }
metricmax = { 'fof2': 14, 'mufd': 30, 'hmf2': 400 }

iriidx = iriidx_table[metric]

def get_data(metric='fof2'):
    with urllib.request.urlopen(f"https://prop.kc2g.com/api/4d_sample?metric={metric}&min_cs=40&count=5000&span=14") as res:
        data = json.loads(res.read().decode())
    return data

def make_dataset(s):
    x, y, cslist = [], [], []
    last_tm = None

    iri = subprocess.Popen(
        ['/build/iri_opt'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )

    for pt in s:
        tm = pt['time']
        dt = pd.to_datetime(tm, unit='s')
        iritime = dt.strftime('%Y %m %d %H %M %S')

        cs = pt['cs']

        meas = pt['meas']
        if meas <= 0 or not jnp.isfinite(meas):
            continue

        iri.stdin.write("%f %f %f %s\n" % (pt['latitude'], pt['longitude'], -100.0, iritime))
        iri.stdin.flush()
        iridata = [float(x) for x in iri.stdout.readline().split()]

        irival = iridata[iriidx]
        if irival <= 0 or not jnp.isfinite(irival):
            continue

        x.append(gen_coords(pt['latitude'], pt['longitude'], tm, dt))
        y.append(jnp.log(meas) - jnp.log(irival))
        cslist.append(cs)
        if last_tm is None or tm > last_tm:
            last_tm = tm

    iri.stdin.close()
    iri.wait()
    ds = {}
    ds['x'] = jax.device_put(jnp.array(x))
    ds['last_tm'] = last_tm
    ds['y'] = jnp.array(y)
    ds['mean'] = jnp.mean(ds['y'])
    ds['y'] -= ds['mean']
    ds['y'] = jax.device_put(ds['y'])
    ds['cs'] = jax.device_put(jnp.array(cslist))

    return ds

iter, fev, best = 0, 0, 999
newbest = False
datasets = []

prev_p = None

def condition(p, x, y, cs):
    cs_int, cs_sl, replace_100, replace_m1 = p[:4]
    kparam = p[4:]

    cs = jnp.where(cs == 100, replace_100, cs)
    cs = jnp.where(cs == -1, replace_m1, cs)

    sigma = cs_int * (1. - cs_sl * cs / 100)
    sigma = jnp.clip(sigma, 1e-3)

    kernel = make_4d_kernel(kparam)
    gp = GaussianProcess(kernel, x, diag=jnp.power(sigma, 2.))
    log_prob, gp_cond = gp.condition(y)
    return (-log_prob / (jnp.shape(x)[0]))

cond_jit = jax.jit(jax.value_and_grad(condition))

def nll(p):
    global fev, best, newbest
    fev = fev + 1

    loss_and_grads = [cond_jit(p, ds['x'], ds['y'], ds['cs']) for ds in datasets]
    lsum = sum([ lg[0] for lg in loss_and_grads ]) / len(loss_and_grads)
    gsum = sum([ lg[1] for lg in loss_and_grads ]) / len(loss_and_grads)

    wloss = -1e20
    wgrad = None

    for lg in loss_and_grads:
        if lg[0] > wloss:
            wloss = lg[0]
            wgrad = lg[1]

    lsum += 0.05 * wloss
    gsum += 0.05 * wgrad

    # if fev % 10 == 0:
    #     print("nll:", fev)

    if lsum < best:
        best = lsum
        newbest = True
    return lsum, gsum

def cb(p):
    global iter, newbest
    iter = iter + 1
    print("# iter=", iter, "fev=", fev, "best=", best, "p=", list(p), "newbest=", newbest)
    newbest = False

def bhcb(p, v, accepted):
    global iter, newbest
    iter = iter + 1
    print("# iter=", iter, "fev=", fev, "this=", v, "best=", best,
          "p=", list(p), "accepted=", accepted, "newbest=", newbest)
    newbest = False

if __name__ == '__main__':
    NDS = 10
    for i in range(NDS):
        print(f"Load {i+1}/{NDS}")
        data = get_data()
        print(f"Process {i+1}/{NDS}")
        ds = make_dataset(data)
        datasets.append(ds)
    print("Done")

    p0 = [
        0.27965685195673856, 0.95, 90.08050977244054, 65.83964838139649,
        -3.973522698703435, -1.4255706715749759, -1.713874124356025,
        -3.7098246830605572, 4.1771638760011545, -0.7053225390616294,
        -4.807779539403842, 6.442619748075302, -1.0820920039196478
    ]

    print("# Init: ", p0)

    bounds = [
        (0.1, None), (0.01, 0.95), (50., 100.), (50., 100.),
        (-20., 20.), (-20., 20.), (-20., 20.),
        (-20., 20.), (-20., 20.), (-20., 20.),
        (-20., 20.), (-20., 20.), (-20., 20.),
    ]

    # opt_result = op.minimize(nll, p0, jac=True, method='TNC', callback=cb, options={'maxfun': 200})
    opt_result = op.minimize(nll, p0, jac=True, method='L-BFGS-B', bounds=bounds,
                             callback=cb, options={'maxfun': 1000 })

    print("# RESULT ", list(opt_result.x))

    # opt_result = namedtuple('OR', ['x'])(p0)

    with urllib.request.urlopen(f"https://prop.kc2g.com/api/4d_sample?metric={metric}&min_cs=25&count=10000&span=4") as res:
        data = json.loads(res.read().decode())
        ds = make_dataset(data)

    cs_int, cs_sl, replace_100, replace_m1 = opt_result.x[:4]
    kparam = opt_result.x[4:]
    kernel = make_4d_kernel(kparam)

    cs = ds['cs']
    cs = jnp.where(cs == 100, replace_100, cs)
    cs = jnp.where(cs == -1, replace_m1, cs)
    sigma = cs_int * (1. - cs_sl * cs / 100)
    sigma = jnp.clip(sigma, 1e-3)

    gp = GaussianProcess(kernel, ds['x'], diag=jnp.power(sigma, 2.))

    for tm in jnp.arange(ds['last_tm'] - 86400, ds['last_tm'] + 2 * 86400 + 1, 3600):
        x = []
        irival = []

        iri = subprocess.Popen(
            ['/build/iri_opt'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )
        dt = pd.to_datetime(tm, unit='s')
        iritime = dt.strftime('%Y %m %d %H %M %S')

        for lat in range(-90, 91):
            for lon in range(-180, 181):
                x.append(gen_coords(lat, lon, tm, dt))
                iri.stdin.write("%f %f %f %s\n" % (lat, lon, -100.0, iritime))
                iri.stdin.flush()
                iridata = [float(x) for x in iri.stdout.readline().split()]
                irival.append(jnp.log(iridata[iriidx]))

        iri.stdin.close()
        iri.wait()

        x = jax.device_put(jnp.array(x))
        irival = jax.device_put(jnp.array(irival))
        p_delta, sd = gp.predict(ds['y'], x, return_var=True)
        p_delta += ds['mean']
        pred = jnp.exp(irival + p_delta)
        sd = jnp.sqrt(sd)

        p_delta = p_delta.reshape((181, 361))
        pred = pred.reshape((181, 361))
        sd = sd.reshape((181, 361))

        fig = plt.figure(figsize=(24, 16))
        plt.imshow(p_delta, extent=(-180, 180, -90, 90), origin='lower', vmin=-.5, vmax=.5)
        plt.tight_layout()
        plt.savefig(f"/out/p_delta_{tm:.0f}.png")
        plt.close(fig)

        fig = plt.figure(figsize=(24, 16))
        plt.imshow(pred, extent=(-180, 180, -90, 90), origin='lower', vmin=0, vmax=metricmax[metric])
        plt.tight_layout()
        plt.savefig(f"/out/pred_{tm:.0f}.png")
        plt.close(fig)

        fig = plt.figure(figsize=(24, 16))
        plt.imshow(sd, extent=(-180, 180, -90, 90), origin='lower', vmin=0., vmax=.6)
        plt.tight_layout()
        plt.savefig(f"/out/sd_{tm:.0f}.png")
        plt.close(fig)
