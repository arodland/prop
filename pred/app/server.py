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

            if False:
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

@app.route("/generate_v2", methods=['POST'])
def generate_v2():
    metrics = [
        { 'name': 'mufd', 'iriidx': 5, 'kernel': delta_kernel },
        { 'name': 'fof2', 'iriidx': 3, 'kernel': delta_kernel },
        { 'name': 'hmf2', 'iriidx': 6, 'kernel': delta_kernel },
    ]

    dsn = "dbname='%s' user='%s' host='%s' password='%s'" % (os.getenv("DB_NAME"), os.getenv("DB_USER"), os.getenv("DB_HOST"), os.getenv("DB_PASSWORD"))
    con = psycopg2.connect(dsn)

    times = [ dt.datetime.fromtimestamp(float(ts)) for ts in request.form.getlist('target') ]
    if len(times) == 0:
        times = [ dt.datetime.utcnow() ]
    predtm = np.array([ time.mktime(ts.timetuple()) for ts in times ])

    run_id = int(request.form.get('run_id', -1))

    data = get_data('http://localhost:%s/history_v2.json?days=14&%s' % (os.getenv('HISTORY_PORT'), '&'.join(['metrics=' + x['name'] for x in metrics])))

    with con.cursor() as cur:
        cur.execute('select ssn from essn where run_id=%s and series=%s order by time desc nulls last limit 1', (run_id, '24h'))
        (ssn,) = cur.fetchone()

    for station in data:
        mout = []
        for mi in range(len(metrics)):
            mout.append({'x': [], 'y': [], 'sigma': []})

        for pt in station['history']:
            tm = pt[0]
            cs = pt[1]
            yval = pt[2+mi]
            if yval is None:
                continue

            if cs < 10 and cs != -1:
                continue

            sd = cs_to_stdev(cs, adj100=True)

            iridata = get_iri(station['latitude'], station['longitude'], ssn, dt.datetime.fromtimestamp(float(tm)))

            for mi, m in enumerate(metrics):
                out = mout[mi]
                yval = pt[2 + mi]
                if yval is not None:
                    y_iri = iridata[m['iriidx']]
                    out['x'].append(tm / 86400.)
                    out['y'].append(np.log(yval) - np.log(y_iri))
                    out['sigma'].append(sd)

        for mi, m in enumerate(metrics):
            out = mout[mi]
            if len(out['x']) < 7:
                continue

            out['x'] = np.array(out['x'])
            out['y'] = np.array(out['y'])
            out['sigma'] = np.array(out['sigma'])
            out['mean'] = np.mean(out['y'])
            out['gp'] = george.GP(m['kernel'])
            out['gp'].compute(out['x'], out['sigma'] + 1e-3)

            pred, sd = out['gp'].predict(out['y'], predtm / 86400., return_var = True)
            out['pred'] = pred
            out['sd'] = sd ** 0.5

        if len([x for x in mout if 'pred' in x]) < 1:
            continue

        for i in range(len(times)):
            iridata = get_iri(station['latitude'], station['longitude'], ssn, times[i])
            col_names, col_vals = [], []
            stdev = None
            for mi, m in enumerate(metrics):
                out = mout[mi]
                if 'pred' not in out:
                    continue
                irival = iridata[m['iriidx']]
                col_names.append(m['name'])
                yout = np.exp(np.log(irival) + out['pred'][i])
                col_vals.append(yout)
                if stdev is None:
                    stdev = out['sd'][i]

            placeholders = ['%s'] * len(col_names)
            upserts = ['{col}=excluded.{col}'.format(col=col) for col in col_names]

            with con.cursor() as cur:
                sql = """
                insert into prediction (run_id, station_id, time, cs, log_stdev, {col_names})
                values (%s, %s, %s, %s, %s, {placeholders})
                on conflict (station_id, run_id, time) do update
                set cs=excluded.cs, log_stdev=excluded.log_stdev,
                {upserts}
                """.format(col_names=", ".join(col_names), placeholders=", ".join(placeholders), upserts=", ".join(upserts))
                cur.execute(sql, (
                    run_id,
                    station['id'],
                    times[i],
                    stdev_to_cs(stdev),
                    stdev,
                    *col_vals
                ))
        con.commit()


    return make_response("OK")

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0', port=int(os.getenv('PRED_PORT')), threaded=False, processes=4)
