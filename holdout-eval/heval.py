from datetime import datetime, timezone, timedelta
import json
import numpy as np
import os
import io
import re
import h5py

import subprocess
import urllib.request
import sys
import multiprocessing

from scipy.interpolate import RectBivariateSpline

import psycopg2

from flask import Flask, request, jsonify

app = Flask(__name__)

def get_holdouts(run_id):
    with urllib.request.urlopen('http://localhost:%s/holdout?run_id=%s' % (os.getenv('API_PORT'), run_id)) as res:
        data = json.loads(res.read().decode())

    for h in data:
        if 'measurement' in h and 'time' in h['measurement']:
            h['measurement']['tm'] = datetime.strptime(h['measurement']['time'], '%Y-%m-%dT%H:%M:%S%z')

    return data

def get_iri(holdout, tm):

    iri = subprocess.Popen(
            ['/build/iri2016_driver', str(tm.year), str(tm.month), str(tm.day), str(tm.hour), str(tm.minute), str(tm.second), str(holdout['station']['latitude']), str(holdout['station']['longitude']), '0', '0', '0'],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            )

    _ = iri.stdout.readline()

    data = [float(x) for x in iri.stdout.readline().split()]
    return {
        'hmf2': data[1],
        'fof2': data[90],
        'mufd': data[90] * data[35],
    }

def read_h5(url):
    with urllib.request.urlopen(url) as res:
        content = res.read()
        bio = io.BytesIO(content)
        h5 = h5py.File(bio, 'r')
        return h5

def spline(table):
    lat = np.linspace(-90, 90, 181)
    lon = np.linspace(-180, 180, 361)
    return RectBivariateSpline(lat, lon, table)

def get_irimap(ds, holdout):
    ret = {}
    lat = float(holdout['station']['latitude'])
    lon = float(holdout['station']['longitude'])
    if lon > 180:
        lon = lon - 360

    for metric in ['fof2', 'hmf2', 'mufd']:
        sp = spline(ds['/maps/' + metric])
        ret[metric] = float(sp(lat, lon, grid=False))

    return ret

def get_all(run_id, holdout, tm):
    irimap = read_h5('http://localhost:%s/irimap.h5?run_id=%s&ts=%d' % (os.getenv('API_PORT'), run_id, tm.timestamp()))
    assimilated = read_h5('http://localhost:%s/assimilated.h5?run_id=%s&ts=%d' % (os.getenv('API_PORT'), run_id, tm.timestamp()))

    meas = holdout['measurement']

    return {
        'station': holdout['station']['code'],
        'holdout_id': holdout['id'],
        'holdout': { 'fof2': meas['fof2'], 'mufd': meas['mufd'], 'hmf2': meas['hmf2'] },
        'iri': get_iri(holdout, tm),
        'irimap': get_irimap(irimap, holdout),
        'assimilated': get_irimap(assimilated, holdout),
    }

def holdout_eval(con, run_id, holdouts):
    out = [ get_all(run_id, holdout, holdout['measurement']['tm']) for holdout in holdouts ]

    for row in out:
        with con.cursor() as cur:
            for model in ['iri', 'irimap', 'assimilated']:
                cur.execute('DELETE FROM holdout_eval WHERE holdout_id=%s and model=%s', (row['holdout_id'], model))
                cur.execute('INSERT INTO holdout_eval (holdout_id, model, fof2, mufd, hmf2) VALUES (%s,%s,%s,%s,%s)', (row['holdout_id'], model, row[model]['fof2'], row[model]['mufd'], row[model]['hmf2']))
                con.commit()
    return out

def pred_eval(con, run_id, holdouts):
    out = []
    for holdout in holdouts:
        with con.cursor() as cur:
            cur.execute('SELECT EXTRACT(epoch FROM a.time) AS ts FROM assimilated a JOIN runs r on a.run_id=r.id WHERE r.id=%s AND a.time >= r.started ORDER BY a.time ASC', (run_id,))
            times = [ int(row[0]) for row in cur.fetchall() ]
            for time in times:
                tm = datetime.fromtimestamp(time, timezone.utc)
                hours_ahead = (time + 300 - times[0]) // 3600
                preds = get_all(run_id, holdout, tm)
                for model in ['iri', 'irimap', 'assimilated']:
                    cur.execute('DELETE FROM pred_eval WHERE holdout_id=%s and model=%s and hours_ahead=%s', (holdout['id'], model, hours_ahead))
                    cur.execute('INSERT INTO pred_eval (holdout_id, model, time, hours_ahead, fof2, mufd, hmf2) VALUES (%s,%s,%s,%s,%s,%s,%s)', (holdout['id'], model, tm, hours_ahead, preds[model]['fof2'], preds[model]['mufd'], preds[model]['hmf2']))
                    con.commit()

    return out

def match_pred_measurements(con):
    with con.cursor() as cur:
        sql = """select pe.id as pred_eval_id, h.id as holdout_id, m.id as measurement_id, pe.hours_ahead, abs(extract(epoch from m.time-pe.time)) as time_diff
        from pred_eval pe
        join holdout h on pe.holdout_id=h.id
        join measurement m on m.time between pe.time - interval '8 minutes' and pe.time + interval '8 minutes'
        and m.station_id=h.station_id
        where pe.measurement_id is null"""
        cur.execute(sql)
        rows = list(cur.fetchall())
        rows.sort(key=lambda row: int(row[3]))
        seen = {}
        for row in rows:
            (pred_eval_id, holdout_id, measurement_id, hours_ahead, time_diff) = row
            if pred_eval_id in seen:
                continue
            seen[pred_eval_id] = time_diff
            cur.execute('UPDATE pred_eval SET measurement_id=%s WHERE id=%s', (measurement_id, pred_eval_id))
            con.commit()

@app.route('/eval', methods=['POST'])
def do_eval_run():
    dsn = "dbname='%s' user='%s' host='%s' password='%s'" % (os.getenv("DB_NAME"), os.getenv("DB_USER"), os.getenv("DB_HOST"), os.getenv("DB_PASSWORD"))
    con = psycopg2.connect(dsn)

    run_id = request.form.get('run_id')
    holdouts = get_holdouts(run_id)
    heval = holdout_eval(con, run_id, holdouts)
    peval = pred_eval(con, run_id, holdouts)
    match_pred_measurements(con)
    return jsonify(heval)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.getenv('HEVAL_PORT')))
