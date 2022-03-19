import os
import io
import sys
from datetime import datetime, timezone

import numpy as np
import h5py

from data import json, jsonapi, hdf5
from models import spline, gp3d, combinators

import psycopg2

from flask import Flask, request, make_response

def get_current():
    return jsonapi.get_data('http://localhost:%s/stations.json' % os.getenv('API_PORT'))

def get_pred(run_id, ts):
    return jsonapi.get_data('http://localhost:%s/pred.json?run_id=%d&ts=%d' % (os.getenv('API_PORT'), run_id, ts))

def get_irimap(run_id, ts):
    return hdf5.get_data('http://localhost:%s/irimap.h5?run_id=%d&ts=%d' % (os.getenv('API_PORT'), run_id, ts))

def get_holdouts(run_id):
    return json.get_data('http://localhost:%s/holdout?run_id=%d' % (os.getenv('API_PORT'), run_id))

def filter_holdouts(df, holdouts):
    if len(holdouts):
        holdout_station_ids = [ row['station']['id'] for row in holdouts ]
        for ii in holdout_station_ids:
            df = df.drop(df[df['station.id'] == ii].index)

    return df

def assimilate(run_id, ts, holdout):
    df_cur = get_current()
    df_pred = get_pred(run_id, ts)
    irimap = get_irimap(run_id, ts)

    if holdout:
        holdouts = get_holdouts(run_id)
        df_pred = filter_holdouts(df_pred, holdouts)

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

    h5.create_dataset('/maps/foe', data=irimap['/maps/foe'])
    h5.create_dataset('/maps/gyf', data=irimap['/maps/gyf'])

    for metric in ["fof2", "hmf2"]:
        df_pred_filtered = jsonapi.filter(df_pred.copy(), required_metrics=[metric], min_confidence=0.1)

        irimodel = spline.Spline(irimap['/maps/' + metric])
        pred = irimodel.predict(df_pred_filtered['station.latitude'].values, df_pred_filtered['station.longitude'].values)

        gp3dmodel = gp3d.GP3D()
        gp3dmodel.train(df_pred_filtered, np.log(df_pred_filtered[metric].values) - np.log(pred))

        model = combinators.Product(irimodel, combinators.LogSpace(gp3dmodel))

        assimilated = model.predict(lat, lon)

        h5.create_dataset('/maps/' + metric, data=assimilated, compression='gzip', scaleoffset=3)
        h5.create_dataset('/stdev/' + metric, data=gp3dmodel.stdev, compression='gzip', scaleoffset=3)

    for metric in ["md"]:
        df_pred_filtered = jsonapi.filter(df_pred.copy(), required_metrics=[metric], min_confidence=0.1)

        iri_mufd = spline.Spline(irimap['/maps/mufd'])
        iri_fof2 = spline.Spline(irimap['/maps/fof2'])

        pred_mufd = iri_mufd.predict(df_pred_filtered['station.latitude'].values, df_pred_filtered['station.longitude'].values)
        pred_fof2 = iri_fof2.predict(df_pred_filtered['station.latitude'].values, df_pred_filtered['station.longitude'].values)

        pred_md = pred_mufd / pred_fof2

        gp3dmodel = gp3d.GP3D()
        gp3dmodel.train(df_pred_filtered, np.log(df_pred_filtered[metric].values) - np.log(pred_md))

        iri_mufd_map = iri_mufd.predict(lat, lon)
        iri_fof2_map = iri_fof2.predict(lat, lon)
        iri_md_map = iri_mufd_map / iri_fof2_map

        log_md_ratio = gp3dmodel.predict(lat, lon)

        assim_md = iri_md_map * np.exp(log_md_ratio)
        h5.create_dataset('/maps/md', data=assim_md, compression='gzip', scaleoffset=3)

        assim_mufd = h5['/maps/fof2'] * assim_md
        h5.create_dataset('/maps/mufd', data=assim_mufd, compression='gzip', scaleoffset=3)

    h5.close()

    return bio.getvalue()

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    dsn = "dbname='%s' user='%s' host='%s' password='%s'" % (os.getenv("DB_NAME"), os.getenv("DB_USER"), os.getenv("DB_HOST"), os.getenv("DB_PASSWORD"))
    con = psycopg2.connect(dsn)

    run_id = int(request.form.get('run_id', -1))
    tgt = int(request.form.get('target', None))
    holdout = bool(request.form.get('holdout', False))

    tm = datetime.fromtimestamp(float(tgt), tz=timezone.utc)

    dataset = assimilate(run_id, tgt, holdout)

    with con.cursor() as cur:
        cur.execute('insert into assimilated (time, run_id, dataset) values (%s, %s, %s) on conflict (run_id, time) do update set dataset=excluded.dataset', (tm, run_id, dataset))
        con.commit()

    con.close()

    return make_response("OK\n")

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.getenv('ASSIMILATE_PORT')), threaded=False, processes=4)
