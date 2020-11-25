import os
import urllib.request, json
import pandas as pd
import numpy as np
import george
from kernel import kernel
from cs import cs_to_stdev, stdev_to_cs
import scipy.optimize as op

from pandas.io.json import json_normalize

def get_data(url=os.getenv("HISTORY_URI")):
    with urllib.request.urlopen(url) as res:
        data = json.loads(res.read().decode())

    return data

data = get_data()
s = data[0]

x, y_fof2, y_mufd, y_hmf2, sigma = [], [], [], [], []
first_tm = None
last_tm = None

for pt in s['history']:
    tm = (pd.to_datetime(pt[0]) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    cs = pt[1]
    if cs < 10 and cs != -1:
        continue

    sd = cs_to_stdev(cs, adj100=True)
    fof2, mufd, hmf2 = pt[2:5]

    print("%d\t%f\t%f\t%f\t%f\t%d" % (tm, fof2, mufd, hmf2, sd, cs))
    x.append(tm / 86400.)
    y_fof2.append(np.log(fof2))
    y_mufd.append(np.log(mufd))
    y_hmf2.append(np.log(hmf2))
    sigma.append(sd)
    if first_tm is None:
        first_tm = tm
    last_tm = tm

print("")

x = np.array(x)
y_fof2 = np.array(y_fof2)
mean_fof2 = np.mean(y_fof2)
y_fof2 -= mean_fof2
y_mufd = np.array(y_mufd)
mean_mufd = np.mean(y_mufd)
y_mufd -= mean_mufd
y_hmf2 = np.array(y_hmf2)
mean_hmf2 = np.mean(y_hmf2)
y_hmf2 -= mean_hmf2
sigma = np.array(sigma)

gp = george.GP(kernel)

iter, fev, grev, best = 0, 0, 0, 999

def nll(p):
    global fev
    global best
    fev = fev + 1
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(y_mufd, quiet=True)
    ll = -ll if np.isfinite(ll) else 1e25
    if ll < best:
        best = ll
    return ll

def grad_nll(p):
    global grev
    grev = grev + 1
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y_mufd, quiet=True)

def cb(p):
    global iter
    iter = iter + 1
    print("# iter=", iter, "fev=", fev, "grev=", grev, "best=", best, "p=", p)

gp.compute(x, sigma + 1e-3)

p0 = gp.get_parameter_vector()
opt_result = op.minimize(nll, p0, jac=grad_nll, method='BFGS', callback=cb, options={'maxiter': 100})
print(opt_result.x)

gp.set_parameter_vector(opt_result.x)

gp.compute(x, sigma + 1e-3)
tm = np.array(range(last_tm - (86400 * 2), last_tm + (86400 * 7) + 1, 300))

pred_fof2, sd_fof2 = gp.predict(y_fof2, tm / 86400., return_var=True)
pred_fof2 += mean_fof2
sd_fof2 = np.sqrt(sd_fof2)
pred_mufd, sd_mufd = gp.predict(y_mufd, tm / 86400., return_var=True)
pred_mufd += mean_mufd
sd_mufd = np.sqrt(sd_mufd)
pred_hmf2, sd_hmf2 = gp.predict(y_hmf2, tm / 86400., return_var=True)
pred_hmf2 += mean_hmf2
sd_hmf2 = np.sqrt(sd_hmf2)

for i in range(len(tm)):
    print("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d" % (
        tm[i],
        np.exp(pred_fof2[i]), np.exp(pred_fof2[i]-sd_fof2[i]), np.exp(pred_fof2[i]+sd_fof2[i]),
        np.exp(pred_mufd[i]), np.exp(pred_mufd[i]-sd_mufd[i]), np.exp(pred_mufd[i]+sd_mufd[i]),
        np.exp(pred_hmf2[i]), np.exp(pred_hmf2[i]-sd_hmf2[i]), np.exp(pred_hmf2[i]+sd_hmf2[i]),
        sd_mufd[i], stdev_to_cs(sd_mufd[i])
    ))

print("")
print("# learned parameters = ", opt_result.x)
print("# learned kernel = ", kernel)
