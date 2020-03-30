import os
import datetime as dt
import time
import copy
import urllib.request, json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import george
from kernel import kernel
from cs import cs_to_stdev, stdev_to_cs

from pandas.io.json import json_normalize

def get_data(url=os.getenv("HISTORY_URI")):
    with urllib.request.urlopen(url) as res:
        data = json.loads(res.read().decode())

    return data

data = get_data()

out = []

for station in data:
    x, y, cslist, sigma = [], [], [], []

    for pt in station['history']:
        tm = (pd.to_datetime(pt[0]) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
        cs = pt[1]
        if cs < 10 and cs != -1:
            continue

        sd = cs_to_stdev(cs, adj100=True)
        fof2, mufd, hmf2 = pt[2:5]

        x.append(tm / 86400.)
        y.append(np.log(mufd))
        cslist.append(cs)
        sigma.append(sd)

    if len(x) < 40:
        continue

    x = np.array(x)
    y = np.array(y)
    ymean = np.mean(y)
    y -= ymean
    cslist = np.array(cslist)
    sigma = np.array(sigma)

    x_train, x_test, y_train, y_test, cs_train, cs_test, sigma_train, sigma_test = train_test_split(
            x, y, cslist, sigma, test_size=0.25)

    gp = george.GP(kernel)
    gp.compute(x_train, sigma_train)

    pred, sd = gp.predict(y_train, x_test, return_var=True)
    sd = sd**0.5

    for i in range(len(pred)):
        print("%d\t%d\t%d\t%f\t%f\t%f" % (
            station['id'],
            x_test[i],
            cs_test[i],
            pred[i],
            y_test[i],
            sd[i],
            )
        )
