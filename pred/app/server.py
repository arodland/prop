from flask import Flask, request, make_response

import os
import datetime as dt
import time
import copy
import urllib.request, json
import pandas as pd
import numpy as np
import george
from kernel import kernel, delta_kernel
from cs import cs_to_stdev, stdev_to_cs
import psycopg2
import subprocess

def get_data(url):
    with urllib.request.urlopen(url) as res:
        data = json.loads(res.read().decode())

    return data

iri = None

def get_iri(lat, lon, ssn, ts):
    global iri

    if iri is None:
        iri = subprocess.Popen(
                ['/build/iri_opt'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                universal_newlines=True,
            )

    iri.stdin.write("%f %f %f %s\n" % (lat, lon, ssn, ts.strftime('%Y %m %d %H %M %S')))
    iri.stdin.flush() 
    data = [float(x) for x in iri.stdout.readline().split()]
    return data # lat, lon, nmf2, fof2, md, mufd, hmf2, foe
                #  0    1    2     3    4    5     6     7 

app = Flask(__name__)

@app.route("/generate", methods=['POST'])
def generate():
    dsn = "dbname='%s' user='%s' host='%s' password='%s'" % (os.getenv("DB_NAME"), os.getenv("DB_USER"), os.getenv("DB_HOST"), os.getenv("DB_PASSWORD"))
    con = psycopg2.connect(dsn)

    times = [ dt.datetime.fromtimestamp(float(ts)) for ts in request.form.getlist('target') ]
    if len(times) == 0:
        times = [ dt.datetime.utcnow() ]

    run_id = int(request.form.get('run_id', -1))

    data = get_data('http://localhost:%s/history.json?days=14' % (os.getenv('HISTORY_PORT')))

    with con.cursor() as cur:
        cur.execute('select ssn from essn where run_id=%s and series=%s order by time desc nulls last limit 1', (run_id, '24h'))
        (ssn,) = cur.fetchone()

    out = []

    for station in data:
        x, y_fof2, y_mufd, y_hmf2, sigma = [], [], [], [], []

        for pt in station['history']:
            ts = pd.to_datetime(pt[0])
            tm = (ts - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
            cs = pt[1]
            if cs < 10 and cs != -1:
                continue

            sd = cs_to_stdev(cs, adj100=True)
            fof2, mufd, hmf2 = pt[2:5]

            iridata = get_iri(station['latitude'], station['longitude'], ssn, ts)
            iri_fof2, iri_mufd, iri_hmf2 = iridata[3], iridata[5], iridata[6]

            x.append(tm / 86400.)
            y_fof2.append(np.log(fof2) - np.log(iri_fof2))
            y_mufd.append(np.log(mufd) - np.log(iri_mufd))
            y_hmf2.append(np.log(hmf2) - np.log(iri_hmf2))
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

        gp = george.GP(delta_kernel)
        gp.compute(x, sigma + 1e-3)

        tm = np.array([ time.mktime(ts.timetuple()) for ts in times ])
        pred_fof2, sd_fof2 = gp.predict(y_fof2, tm / 86400., return_var=True)
        pred_fof2 += mean_fof2
        sd_fof2 = sd_fof2**0.5
        pred_mufd, sd_mufd = gp.predict(y_mufd, tm / 86400., return_var=True)
        pred_mufd += mean_mufd
        sd_mufd = sd_mufd**0.5
        pred_hmf2, sd_hmf2 = gp.predict(y_hmf2, tm / 86400., return_var=True)
        pred_hmf2 += mean_hmf2
        sd_hmf2 = sd_hmf2**0.5

        for i in range(len(times)):
            iridata = get_iri(station['latitude'], station['longitude'], ssn, times[i])
            iri_fof2, iri_mufd, iri_hmf2 = iridata[3], iridata[5], iridata[6]
            fof2 = np.exp(np.log(iri_fof2) + pred_fof2[i])
            mufd = np.exp(np.log(iri_mufd) + pred_mufd[i])
            hmf2 = np.exp(np.log(iri_hmf2) + pred_hmf2[i])

            delta_fof2 = fof2 - iri_fof2
            delta_mufd = mufd - iri_mufd
            delta_hmf2 = hmf2 - iri_hmf2

            with con.cursor() as cur:
                cur.execute("""
                insert into prediction (run_id, station_id, time, cs, log_stdev, fof2, mufd, hmf2) 
                values (%s, %s, %s, %s, %s, %s, %s, %s)
                on conflict (station_id, run_id, time) do update
                set cs=excluded.cs, log_stdev=excluded.log_stdev,
                fof2=excluded.fof2, mufd=excluded.mufd, hmf2=excluded.hmf2
                """,
                    (
                        run_id,
                        station['id'],
                        times[i],
                        stdev_to_cs(sd_mufd[i]),
                        sd_mufd[i],
                        fof2,
                        mufd,
                        hmf2,
                    )
                )

            with con.cursor() as cur:
                cur.execute("""
                insert into prediction_delta (run_id, station_id, time, cs, log_stdev, delta_fof2, delta_mufd, delta_hmf2) 
                values (%s, %s, %s, %s, %s, %s, %s, %s)
                on conflict (station_id, run_id, time) do update
                set cs=excluded.cs, log_stdev=excluded.log_stdev,
                delta_fof2=excluded.delta_fof2, delta_mufd=excluded.delta_mufd, delta_hmf2=excluded.delta_hmf2
                """,
                    (
                        run_id,
                        station['id'],
                        times[i],
                        stdev_to_cs(sd_mufd[i]),
                        sd_mufd[i],
                        delta_fof2,
                        delta_mufd,
                        delta_hmf2,
                    )
                )

        con.commit()


    return make_response("OK")

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0', port=int(os.getenv('PRED_PORT')), threaded=False, processes=4)
