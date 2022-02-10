from datetime import datetime, timezone, timedelta
import igrf
import json
import numpy as np
import os
import re
from scipy.optimize import minimize_scalar

import subprocess
import urllib.request
import sys
import multiprocessing

from scipy import interpolate
from scipy.signal import windows

import psycopg2

from flask import Flask, request, jsonify

def station_weight(station, tm):
    hour = float(tm.hour) + float(tm.minute) / 60. + float(tm.second) / 3600.
    local_time = (hour + float(station['longitude']) / 15.) % 24.
    abs_dip = abs(station['dip_angle'])

    time_weight = np.sqrt(2 + np.cos((local_time - 14.5) * np.pi / 12.)) - 0.73
    if abs_dip < 25.:
        return 0.75 * time_weight
    elif abs_dip < 50.:
        return 1. * time_weight
    elif abs_dip < 56.:
        return 0.75 * time_weight
    else:
        return 0.5 * time_weight

tukey_int = interpolate.interp1d(
    np.linspace(0, 25, 2501),
    windows.tukey(2501, alpha=0.08),
)

def recency_weight(tm, now, recency):
    delta = now - tm
    hours = delta / timedelta(hours=1)

    if hours < 0:
        return 0

    if recency:
        if hours >= 24:
            return 0
        else:
            return np.power(2.0, -(hours/6))
    else:
        if hours >= 25:
            return 0
        else:
            return tukey_int(hours)

def cs_to_stdev(cs):
    if cs == -1 or cs > 100:
        cs = 62
    if cs == 100:
        cs = 86
    return 0.237 - 0.00170 * cs

iri = None
pred_pool = None

def get_pred_station(station, ssn, iritime):
    return get_pred((station['latitude'], station['longitude'], ssn, iritime))

def get_pred(params):
    global iri
    if iri is None:
        iri = subprocess.Popen(
                ['/build/iri_opt'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                universal_newlines=True,
                )

    iri.stdin.write("%f %f %f %s\n" % params)
    iri.stdin.flush()
    data = [float(x) for x in iri.stdout.readline().split()]
    return data # lat, lon, nmf2, fof2, md, mufd, hmf2, foe
                #  0    1    2     3    4    5     6     7 

def station_err(x):
    station, ssn, recency, now = x

    station_total = 0.0
    station_tw = 0.0
    num_records = 0

    if len(station['history']) < 3:
        return (0,0)

    last_tm = station['history'][-1][5]
    if (now - last_tm).total_seconds() > 7200:
        return (0,0)

    for record in station['history']:
        ts, cs, fof2, mufd, hmf2, tm, iritime = record

        if cs <= 25:
            continue

        pred = get_pred_station(station, ssn, iritime)
        fof2_pred, hmf2_pred, mufd_pred = pred[3], pred[6], pred[5]
        stdev = cs_to_stdev(cs)

        err = 0.6 * np.abs(np.log(fof2_pred) - np.log(fof2)) / 0.34873
        err += 0.3 * np.abs(np.log(mufd_pred) - np.log(mufd)) / 0.37516
        err += 0.1 * np.abs(np.log(hmf2_pred) - np.log(hmf2)) / 0.18405
        err /= stdev

        sw = station_weight(station, tm)
        rw = recency_weight(tm, now, recency)
        weight = sw * rw

        station_total += err * weight
        station_tw += weight
        num_records += 1

    if num_records >= 3:
        return (station_total / float(num_records), station_tw / float(num_records))
    else:
        return (0,0)


def err(ssn, data, recency, now):
    global pred_pool

    total = 0.0
    total_weight = 0.0

    if pred_pool is None:
        pred_pool = multiprocessing.Pool(16)

    results = pred_pool.imap_unordered(station_err, [ (station, ssn, recency, now) for station in data ])
    for result in results:
        total += result[0]
        total_weight += result[1]

    return total / total_weight

def get_holdouts(run_id, num):
    urllib.request.urlopen('http://localhost:%s/holdout' % os.getenv('API_PORT'), data=urllib.parse.urlencode({'run_id': str(run_id), 'num': str(num)}).encode('ascii'))
    with urllib.request.urlopen('http://localhost:%s/holdout?run_id=%s' % (os.getenv('API_PORT'), run_id)) as res:
        data = json.loads(res.read().decode())

    return data

def generate_essn(run_id, series, num_holdouts):
    now = datetime.utcnow()
    recency = True if series == '6h' else False

    dsn = "dbname='%s' user='%s' host='%s' password='%s'" % (os.getenv("DB_NAME"), os.getenv("DB_USER"), os.getenv("DB_HOST"), os.getenv("DB_PASSWORD"))
    con = psycopg2.connect(dsn)

    with urllib.request.urlopen('http://localhost:%s/history.json?days=2' % (os.getenv('HISTORY_PORT'))) as res:
        data = json.loads(res.read().decode())

    data = [ station for station in data if station['use_for_essn'] == 1 ]

    if num_holdouts > 0:
        holdouts = get_holdouts(run_id, num_holdouts)
        exclude_station_ids = [ row['station']['id'] for row in holdouts ]
        data = [ station for station in data if station['id'] not in exclude_station_ids ]

    for station in data:
        # TODO: use the current time once IGRF13 is available
        mag = igrf.igrf(
                now.strftime('%Y-%m-%d'),
                glat=station['latitude'],
                glon=station['longitude'],
                alt_km=1.
                )

        station['dip_angle'] = mag['incl'].item()

        for record in station['history']:
            record[0] = re.sub(r"\.\d+$", "", record[0])
            record.append(datetime.strptime(record[0], '%Y-%m-%d %H:%M:%S'))
            record.append(datetime.strftime(record[5], '%Y %m %d %H %M %S'))


    res = minimize_scalar(err, args=(data, recency, now), bounds=(-20.0, 200.0), method='Bounded', options={'xatol':0.01, 'maxiter': 1000})
    ssn = res.x
    sfi = 63.75 + ssn * (0.728 + ssn*0.000089)

    cur = con.cursor()

    cur.execute(
        'INSERT INTO essn (time, series, run_id, sfi, ssn, err) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (series, run_id, time) DO UPDATE SET sfi=excluded.sfi, ssn=excluded.ssn, err=excluded.err',
        (now, series, int(run_id), sfi, ssn, res.fun)
    )

    con.commit()
    cur.close()
    con.close()

    return { 'sfi': sfi, 'ssn': ssn, 'err': res.fun }


app = Flask(__name__)
@app.route('/generate', methods=['POST'])
def generate():
    run_id = request.form.get('run_id', -1)
    series = request.form.get('series', '24h')
    num_holdouts = int(request.form.get('num_holdouts', 0))

    return jsonify(generate_essn(run_id, series, num_holdouts))

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.getenv('ESSN_PORT')))
