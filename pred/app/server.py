from flask import Flask, request, jsonify, render_template

import os
import datetime as dt
import time
import copy
import urllib.request, json
import pandas as pd
import numpy as np
import george
from kernel import kernel
from cs import cs_to_stdev, stdev_to_cs

from pandas.io.json import json_normalize

def get_data(url=os.getenv("HISTORY_URI")):
    with urllib.request.urlopen(url) as res:
        data = json.loads(res.read().decode())

    return data

app = Flask(__name__)

@app.route("/stations.json", methods=['GET'])
def stationsjson():
    ts = request.args.get('time', None)
    now = dt.datetime.utcnow()
    if ts is not None:
        now = dt.datetime.fromtimestamp(float(ts))

    data = get_data()

    out = []

    for station in data:
        x, y_fof2, y_mufd, y_hmf2, sigma = [], [], [], [], []

        for pt in station['history']:
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

        if len(x) < 7:
            continue

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
        gp.compute(x, 2. * sigma + 1e-4)

        tm = np.array([ time.mktime(now.timetuple()) ])
        pred_fof2, sd_fof2 = gp.predict(y_fof2, tm / 86400., return_var=True)
        pred_fof2 += mean_fof2
        sd_fof2 = sd_fof2**0.5
        pred_mufd, sd_mufd = gp.predict(y_mufd, tm / 86400., return_var=True)
        pred_mufd += mean_mufd
        sd_mufd = sd_mufd**0.5
        pred_hmf2, sd_hmf2 = gp.predict(y_hmf2, tm / 86400., return_var=True)
        pred_hmf2 += mean_hmf2
        sd_hmf2 = sd_hmf2**0.5

        st_out = {
            "station": copy.copy(station),
            "cs": stdev_to_cs(sd_mufd[0]),
            "fof2": round(np.exp(pred_fof2[0]), 2),
            "mufd": round(np.exp(pred_mufd[0]), 2),
            "hmf2": round(np.exp(pred_hmf2[0]), 2),
            "time": now.strftime("%Y-%m-%d %H:%M:%S")
        }
        del(st_out["station"]["history"])

        out.append(st_out)

    return jsonify(out)

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0', port=5000)
