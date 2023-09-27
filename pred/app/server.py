from flask import Flask, request, make_response

import os
import datetime as dt
import time
import copy
import urllib.request, json
import pandas as pd
import numpy as np
import george
from kernel import kernel, delta_kernel, delta_fof2_kernel, delta_hmf2_kernel, delta_mufd_kernel
from cs import cs_to_stdev, cs_to_stdev_new, stdev_to_cs
import psycopg
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
    kernel_choice = request.form.get('kernels', 'old')
    if kernel_choice == 'new':
        metrics = [
            { 'name': 'mufd', 'iriidx': 5, 'kernel': delta_mufd_kernel, 'cs': 'mufd' },
            { 'name': 'fof2', 'iriidx': 3, 'kernel': delta_fof2_kernel, 'cs': 'fof2' },
            { 'name': 'hmf2', 'iriidx': 6, 'kernel': delta_hmf2_kernel, 'cs': 'hmf2' },
        ]
    else:
        metrics = [
            { 'name': 'mufd', 'iriidx': 5, 'kernel': delta_kernel, 'cs': 'old' },
            { 'name': 'fof2', 'iriidx': 3, 'kernel': delta_kernel, 'cs': 'old' },
            { 'name': 'hmf2', 'iriidx': 6, 'kernel': delta_kernel, 'cs': 'old' },
        ]

    dsn = "dbname='%s' user='%s' host='%s' password='%s'" % (os.getenv("DB_NAME"), os.getenv("DB_USER"), os.getenv("DB_HOST"), os.getenv("DB_PASSWORD"))
    con = psycopg.connect(dsn)

    station = request.form.get('station', None)

    times = [ dt.datetime.fromtimestamp(float(ts)) for ts in request.form.getlist('target') ]
    if len(times) == 0:
        times = [ dt.datetime.utcnow() ]
    predtm = np.array([ time.mktime(ts.timetuple()) for ts in times ])

    run_id = int(request.form.get('run_id', -1))

    station_q = '' if station is None else 'station=%d&' % int(station)
    data = get_data('http://localhost:%s/history_v2.json?days=14&%s%s' % (os.getenv('HISTORY_PORT'), station_q, '&'.join(['metrics=' + x['name'] for x in metrics])))

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

            iridata = get_iri(station['latitude'], station['longitude'], ssn, dt.datetime.fromtimestamp(float(tm)))

            for mi, m in enumerate(metrics):
                out = mout[mi]
                sd = cs_to_stdev_new(cs, m['cs'], adj100=True)
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
            col_names, col_vals, stdev_names, stdev_vals = [], [], [], []
            stdev = None
            for mi, m in enumerate(metrics):
                out = mout[mi]
                if 'pred' not in out:
                    continue
                irival = iridata[m['iriidx']]
                col_names.append(m['name'])
                stdev_names.append('stdev_'+m['name'])
                yout = np.exp(np.log(irival) + out['pred'][i])
                col_vals.append(yout)
                stdev_vals.append(out['sd'][i])

                if stdev is None:
                    stdev = out['sd'][i]

            print('run_id: {run_id} time: {time} stdev_vals: {stdev_vals}'.format(run_id=run_id, time=time, stdev_vals=stdev_vals))

            placeholders = ['%s'] * len(col_names)
            upserts = ['{col}=excluded.{col}'.format(col=col) for col in col_names + stdev_names]

            with con.cursor() as cur:
                sql = """
                insert into prediction (run_id, station_id, time, cs, {stdev_names}, {col_names})
                values (%s, %s, %s, %s, {placeholders}, {placeholders})
                on conflict (station_id, run_id, time) do update
                set cs=excluded.cs,
                {upserts}
                """.format(col_names=", ".join(col_names), stdev_names=", ".join(stdev_names), placeholders=", ".join(placeholders), upserts=", ".join(upserts))
                cur.execute(sql, (
                    run_id,
                    station['id'],
                    times[i],
                    stdev_to_cs(stdev),
                    *stdev_vals,
                    *col_vals
                ))
        con.commit()


    return make_response("OK")
