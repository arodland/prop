import os
import io
import gzip
import struct
from datetime import datetime, timezone

import numpy as np
import h5py
import hdf5plugin

import wquantiles

from data import json, jsonapi, hdf5
from models import spline, gp3d, combinators

import psycopg

from flask import Flask, request, make_response

def get_current():
    return jsonapi.get_data('http://localhost:%s/stations.json' % os.getenv('API_PORT'))

def get_pred(run_id, ts):
    return jsonapi.get_data('http://localhost:%s/pred.json?run_id=%d&ts=%d' % (os.getenv('API_PORT'), run_id, ts))

def get_irimap(run_id, ts):
    return hdf5.get_data('http://localhost:%s/irimap.h5?run_id=%d&ts=%d' % (os.getenv('API_PORT'), run_id, ts))

def get_ipe(run_id, ts):
    return hdf5.get_data('http://localhost:%s/ipe.h5?run_id=%d&ts=%d' % (os.getenv('API_PORT'), run_id, ts))

def get_assimilated(run_id, ts):
    return hdf5.get_data('http://localhost:%s/assimilated.h5?run_id=%d&ts=%d' % (os.getenv('API_PORT'), run_id, ts))

def get_holdouts(run_id):
    return json.get_data('http://localhost:%s/holdout?run_id=%d' % (os.getenv('API_PORT'), run_id))

def filter_holdouts(df, holdouts):
    if len(holdouts):
        holdout_station_ids = [ row['station']['id'] for row in holdouts ]
        for ii in holdout_station_ids:
            df = df.drop(df[df['station.id'] == ii].index)

    return df

def get_scale(base, target, minval):
    base_trimmed = base[1:180, 0:360].flatten()
    target_trimmed = target[1:180, 0:360].flatten()
    weight = np.repeat(np.cos(np.linspace(-np.pi / 2, np.pi / 2, 181)[1:180]), 360) # cos(latitude)

    base_med = wquantiles.quantile(base_trimmed, weight, 0.5)
    target_med = wquantiles.quantile(target_trimmed, weight, 0.5)
    target_min = target_trimmed.min()

    iqr_ratio = (
        (wquantiles.quantile(base_trimmed, weight, 0.75) - wquantiles.quantile(base_trimmed, weight, 0.25)) /
        (wquantiles.quantile(target_trimmed, weight, 0.75) - wquantiles.quantile(target_trimmed, weight, 0.25))
    )

    # Clamp iqr_ratio to never produce a value < minval.
    if (target_min - target_med) * iqr_ratio + base_med < minval:
        iqr_ratio = (base_med - minval) / (target_med - target_min)

    return (base_med, target_med, iqr_ratio)

def rescale(target, params):
    (base_med, target_med, iqr_ratio) = params
    return (target - target_med) * iqr_ratio + base_med

def assimilate(run_id, ts, holdout, basemap_type, cs_type):
    df_cur = get_current()
    df_pred = get_pred(run_id, ts)
    irimap = get_irimap(run_id, ts)

    if holdout:
        holdouts = get_holdouts(run_id)
        df_pred = filter_holdouts(df_pred, holdouts)

    if basemap_type == 'iri':
        basemap = irimap
    else:
        ipemap = get_ipe(run_id, ts)
        ipemap = {k: ipemap[k] for k in ('/maps/hmf2', '/maps/fof2', '/maps/mufd')}

        if basemap_type.endswith('_scaled'):
            basemap_type = basemap_type[:len(basemap_type) - len('_scaled')]
            fof2_scale = get_scale(irimap['/maps/fof2'], ipemap['/maps/fof2'], 0.5)
            ipemap['/maps/fof2'] = rescale(ipemap['/maps/fof2'], fof2_scale)
            ipemap['/maps/mufd'] = rescale(ipemap['/maps/mufd'], fof2_scale)
        elif basemap_type.endswith('_logscaled'):
            basemap_type = basemap_type[:len(basemap_type) - len('_logscaled')]
            fof2_scale = get_scale(np.log(irimap['/maps/fof2']), np.log(ipemap['/maps/fof2']), np.log(0.5))
            new_fof2 = np.exp(rescale(np.log(ipemap['/maps/fof2']), fof2_scale))
            fof2_ratio = new_fof2 / ipemap['/maps/fof2'][:]
            ipemap['/maps/fof2'] = new_fof2
            ipemap['/maps/mufd'] = ipemap['/maps/mufd'][:] * fof2_ratio

        if basemap_type == 'ipe':
            basemap = ipemap
        elif basemap_type == 'iri-ipe':
            basemap = {k: (irimap[k][:] + ipemap[k][:]) / 2.0 for k in ('/maps/hmf2', '/maps/fof2', '/maps/mufd')}

    bio = io.BytesIO()
    h5 = h5py.File(bio, 'w')

    lat, lon = np.meshgrid(
        np.linspace(-90, 90, 181),
        np.linspace(-180, 180, 361),
        indexing='ij',
    )

    h5.create_dataset('/essn/ssn', data=irimap['/essn/ssn'])
    h5.create_dataset('/essn/sfi', data=irimap['/essn/sfi'])
    h5.create_dataset('/ts', data=irimap['/ts'])
    h5.create_dataset('/stationdata/curr', data=df_cur.to_json(orient='records'))
    h5.create_dataset('/stationdata/pred', data=df_pred.to_json(orient='records'))

    h5.create_dataset('/maps/foe', data=irimap['/maps/foe'], **hdf5plugin.SZ(absolute=0.001))
    h5.create_dataset('/maps/gyf', data=irimap['/maps/gyf'], **hdf5plugin.SZ(absolute=0.001))

    for metric in ["fof2", "hmf2"]:
        df_pred_filtered = jsonapi.filter(df_pred.copy(), required_metrics=[metric], min_confidence=0.1)

        basemodel = spline.Spline(basemap['/maps/' + metric])
        pred = basemodel.predict(df_pred_filtered['station.latitude'].values,
                                 df_pred_filtered['station.longitude'].values)

        if cs_type == 'new':
            stdev = df_pred_filtered['stdev_' + metric].values
        else:
            stdev = 0.203 - 0.170 * df_pred_filtered.cs.values

        gp3dmodel = gp3d.GP3D()
        gp3dmodel.train(df_pred_filtered, np.log(df_pred_filtered[metric].values) - np.log(pred), stdev)

        model = combinators.Product(basemodel, combinators.LogSpace(gp3dmodel))

        assimilated = model.predict(lat, lon)

        h5.create_dataset('/maps/' + metric, data=assimilated, **hdf5plugin.SZ(absolute=0.001))
        h5.create_dataset('/stdev/' + metric, data=gp3dmodel.stdev, **hdf5plugin.SZ(absolute=0.0001))

    for metric in ["md"]:
        df_pred_filtered = jsonapi.filter(df_pred.copy(), required_metrics=[metric], min_confidence=0.1)

        base_mufd = spline.Spline(basemap['/maps/mufd'])
        base_fof2 = spline.Spline(basemap['/maps/fof2'])

        pred_mufd = base_mufd.predict(df_pred_filtered['station.latitude'].values,
                                      df_pred_filtered['station.longitude'].values)
        pred_fof2 = base_fof2.predict(df_pred_filtered['station.latitude'].values,
                                      df_pred_filtered['station.longitude'].values)

        pred_md = pred_mufd / pred_fof2

        if cs_type == 'new':
            stdev = df_pred_filtered['stdev_' + metric].values
        else:
            stdev = 0.203 - 0.170 * df_pred_filtered.cs.values

        gp3dmodel = gp3d.GP3D()
        gp3dmodel.train(df_pred_filtered, np.log(df_pred_filtered[metric].values) - np.log(pred_md), stdev)

        base_mufd_map = base_mufd.predict(lat, lon)
        base_fof2_map = base_fof2.predict(lat, lon)
        base_md_map = base_mufd_map / base_fof2_map

        log_md_ratio = gp3dmodel.predict(lat, lon)

        assim_md = base_md_map * np.exp(log_md_ratio)
        h5.create_dataset('/maps/md', data=assim_md, **hdf5plugin.SZ(absolute=0.001))

        assim_mufd = h5['/maps/fof2'] * assim_md
        h5.create_dataset('/maps/mufd', data=assim_mufd, **hdf5plugin.SZ(absolute=0.001))

    h5.close()

    return bio.getvalue()

def iongrid(run_id, tss):
    models = [ {} for hour in range(24) ]

    for ts in tss:
        uthour = (ts % 86400) // 3600
        ds = get_assimilated(run_id, ts)
        models[uthour] = {
            'fof2': ds['/maps/fof2'],
            'md': ds['/maps/md'],
        }

    bio = io.BytesIO()
    gz = gzip.open(bio, mode='wb')

    for metric in ['fof2', 'md']:
        for hour in range(24):
            model = models[hour][metric]
            for lati in range(181):
                for loni in range(361):
                    gz.write(struct.pack('<f', model[lati, loni]))

    gz.close()
    return bio.getvalue()

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    dsn = "dbname='%s' user='%s' host='%s' password='%s'" % (
        os.getenv("DB_NAME"), os.getenv("DB_USER"), os.getenv("DB_HOST"), os.getenv("DB_PASSWORD"))
    con = psycopg.connect(dsn)

    run_id = int(request.form.get('run_id', -1))
    tgt = int(request.form.get('target', None))
    holdout = bool(request.form.get('holdout', False))
    basemap_type = request.form.get('basemap', 'iri')
    cs_type = request.form.get('cs', 'old')

    tm = datetime.fromtimestamp(float(tgt), tz=timezone.utc)

    dataset = assimilate(run_id, tgt, holdout, basemap_type, cs_type)

    with con.cursor() as cur:
        cur.execute("""insert into assimilated (time, run_id, dataset)
                    values (%s, %s, %s)
                    on conflict (run_id, time) do update set dataset=excluded.dataset""",
                    (tm, run_id, dataset)
                    )
        con.commit()

    con.close()

    return make_response("OK\n")

@app.route('/generate_iongrid', methods=['POST'])
def generate_iongrid():
    dsn = "dbname='%s' user='%s' host='%s' password='%s'" % (
        os.getenv("DB_NAME"), os.getenv("DB_USER"), os.getenv("DB_HOST"), os.getenv("DB_PASSWORD"))
    con = psycopg.connect(dsn)

    run_id = int(request.form.get('run_id', -1))
    tgts = [ int(ts) for ts in request.form.getlist('target') ]

    dataset = iongrid(run_id, tgts)

    with con.cursor() as cur:
        cur.execute("""insert into iongrid (run_id, dataset)
                    values (%s, %s)
                    on conflict (run_id) do update set dataset=excluded.dataset""",
                    (run_id, dataset)
                    )
        con.commit()

    con.close()

    return make_response("OK\n")

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.getenv('ASSIMILATE_PORT')), threaded=False, processes=4)
