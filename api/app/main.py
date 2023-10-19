from flask import Flask, request, jsonify, render_template, send_file, make_response, redirect, Response, stream_with_context
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from marshmallow import Schema, fields
from sqlalchemy import tablesample
from sqlalchemy.orm import aliased
import json
import os
import io
import datetime as dt
import dateutil
from sqlalchemy import and_, text
import urllib.request
import maidenhead
from pymemcache.client.base import Client as MemcacheClient
import re
import logging
import pandas as pd
import pyarrow as pa

#logging.basicConfig()
#logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
from werkzeug.middleware.profiler import ProfilerMiddleware

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql+psycopg://%s:%s@%s:5432/%s' % (
    os.getenv("DB_USER"), os.getenv("DB_PASSWORD"), os.getenv("DB_HOST"), os.getenv("DB_NAME"))
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# app.wsgi_app = ProfilerMiddleware(app.wsgi_app, profile_dir='/profiles', stream=None)

# Order matters: Initialize SQLAlchemy before Marshmallow
db = SQLAlchemy(app)
ma = Marshmallow(app)

memcache = MemcacheClient('127.0.0.1', ignore_exc=True, no_delay=True)

#Declare models 

@stream_with_context
def dump_streaming(obj, schema):
    yield "["
    it = iter(obj)
    i = next(it, None)
    while i is not None:
        yield schema.dumps(i)
        i = next(it, None)
        if i is not None:
            yield ","
    yield "]"

@stream_with_context
def arrow_streaming(qry, con, remove_fields=[]):
    dfi = pd.read_sql(qry, con, chunksize=250000, dtype_backend='pyarrow')
    bio = io.BytesIO()
    it = iter(dfi)
    i = next(it, None)
    if i is not None:
        schema = pa.Schema.from_pandas(i)
        for field in remove_fields:
            schema = schema.remove(schema.get_field_index(field))
        writer = pa.ipc.new_stream(bio, schema, options=pa.ipc.IpcWriteOptions(compression='lz4'))
    while i is not None:
        batch = pa.RecordBatch.from_pandas(i, schema=schema, preserve_index=False)
        writer.write_batch(batch)
        yield bio.getvalue() + b""
        bio.seek(0)
        bio.truncate(0)
        i = next(it, None)
    writer.close()
    yield bio.getvalue()

class Station(db.Model):
    id = db.Column(db.Integer,  primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    code = db.Column(db.String, unique=True)
    longitude = db.Column(db.Text)
    latitude = db.Column(db.Text)
    use_for_essn = db.Column(db.Boolean)
    use_for_maps = db.Column(db.Boolean)
    measurements = db.relationship('Measurement', back_populates='station')

    def __repr__(self):
        return '<Station %r>' % self.name

class Measurement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.DateTime)
    cs = db.Column(db.Numeric(asdecimal=False))
    fof2 = db.Column(db.Numeric(asdecimal=False))
    fof1 = db.Column(db.Numeric(asdecimal=False))
    mufd = db.Column(db.Numeric(asdecimal=False))
    foes = db.Column(db.Numeric(asdecimal=False))
    foe = db.Column(db.Numeric(asdecimal=False))
    hf2 = db.Column(db.Numeric(asdecimal=False))
    he = db.Column(db.Numeric(asdecimal=False))
    hme = db.Column(db.Numeric(asdecimal=False))
    hmf2 = db.Column(db.Numeric(asdecimal=False))
    hmf1 = db.Column(db.Numeric(asdecimal=False))
    yf2 = db.Column(db.Numeric(asdecimal=False))
    yf1 = db.Column(db.Numeric(asdecimal=False))
    tec = db.Column(db.Numeric(asdecimal=False))
    scalef2 = db.Column(db.Numeric(asdecimal=False))
    fbes = db.Column(db.Numeric(asdecimal=False))
    source = db.Column(db.Text)
    station_id = db.Column(db.Integer, db.ForeignKey('station.id'))
    station = db.relationship('Station', foreign_keys=[station_id], back_populates='measurements')
    md = db.Column(db.Text)
    def __repr__(self):
        return '<Measurement %r>' % self.id


class Runs(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    started = db.Column(db.DateTime)
    ended = db.Column(db.DateTime)
    target_time = db.Column(db.DateTime)
    state = db.Enum('created', 'finished', 'archived', 'deleted', 'uploaded', name='run_state')
    experiment = db.Column(db.Text)
    def __repr__(self):
        return '<Run %r>' % self.id


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.Integer)
    time = db.Column(db.DateTime)
    cs = db.Column(db.Numeric(asdecimal=False))
    fof2 = db.Column(db.Numeric(asdecimal=False))
    mufd = db.Column(db.Numeric(asdecimal=False))
    hmf2 = db.Column(db.Numeric(asdecimal=False))
    stdev_fof2 = db.Column(db.Numeric(asdecimal=False))
    stdev_mufd = db.Column(db.Numeric(asdecimal=False))
    stdev_hmf2 = db.Column(db.Numeric(asdecimal=False))
    station_id = db.Column(db.Integer, db.ForeignKey('station.id'))
    station = db.relationship('Station', foreign_keys=[station_id])
    def __repr__(self):
        return '<Prediction %r>' % self.id

class Holdout(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.Integer, db.ForeignKey('runs.id'))
    run = db.relationship('Runs', foreign_keys=[run_id], lazy='joined', innerjoin=True)
    station_id = db.Column(db.Integer, db.ForeignKey('station.id'))
    station = db.relationship('Station', foreign_keys=[station_id], lazy='joined', innerjoin=True)
    measurement_id = db.Column(db.Integer, db.ForeignKey('measurement.id'))
    measurement = db.relationship('Measurement', foreign_keys=[measurement_id], lazy='joined', innerjoin=True)
    def __repr__(self):
        return '<Holdout %r: %r %r %r>' % (self.id, self.run_id, self.station_id, self.measurement_id)

class HoldoutEval(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    holdout_id = db.Column(db.Integer, db.ForeignKey('holdout.id'))
    holdout = db.relationship('Holdout', foreign_keys=[holdout_id], lazy='joined', innerjoin=True)
    model = db.Column(db.Text)
    fof2 = db.Column(db.Numeric(asdecimal=False))
    mufd = db.Column(db.Numeric(asdecimal=False))
    hmf2 = db.Column(db.Numeric(asdecimal=False))

class PredEval(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    holdout_id = db.Column(db.Integer, db.ForeignKey('holdout.id'))
    holdout = db.relationship('Holdout', foreign_keys=[holdout_id], lazy='joined', innerjoin=True)
    model = db.Column(db.Text)
    time = db.Column(db.DateTime)
    hours_ahead = db.Column(db.Integer)
    fof2 = db.Column(db.Numeric(asdecimal=False))
    mufd = db.Column(db.Numeric(asdecimal=False))
    hmf2 = db.Column(db.Numeric(asdecimal=False))
    measurement_id = db.Column(db.Integer, db.ForeignKey('measurement.id'))
    measurement = db.relationship('Measurement', foreign_keys=[measurement_id], lazy='joined', innerjoin=True)

class CosmicEval(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.DateTime)
    run_id = db.Column(db.Integer, db.ForeignKey('runs.id'))
    run = db.relationship('Runs', foreign_keys=[run_id])
    hours_ahead = db.Column(db.Integer)
#    source = db.Enum('cosmic-2', 'planetiq', name='cosmic_source')
    source = db.Column(db.Text)
    nearest_station = db.Column(db.Integer, db.ForeignKey('station.id'))
    station_distance = db.Column(db.Numeric(asdecimal=False))
    latitude = db.Column(db.Numeric(asdecimal=False))
    longitude = db.Column(db.Numeric(asdecimal=False))
    fof2_true = db.Column(db.Numeric(asdecimal=False))
    fof2_iri = db.Column(db.Numeric(asdecimal=False))
    fof2_irimap = db.Column(db.Numeric(asdecimal=False))
    fof2_full = db.Column(db.Numeric(asdecimal=False))
    fof2_irtam = db.Column(db.Numeric(asdecimal=False))
    hmf2_true = db.Column(db.Numeric(asdecimal=False))
    hmf2_iri = db.Column(db.Numeric(asdecimal=False))
    hmf2_irimap = db.Column(db.Numeric(asdecimal=False))
    hmf2_full = db.Column(db.Numeric(asdecimal=False))
    hmf2_irtam = db.Column(db.Numeric(asdecimal=False))
    dip_angle = db.Column(db.Numeric(asdecimal=False))
    modip = db.Column(db.Numeric(asdecimal=False))
    generation = db.Column(db.Integer)

#Generate marshmallow Schemas from your models using ModelSchema

class StationSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Station # Fields to expose
        load_instance = True

station_schema = StationSchema()
stations_schema = StationSchema(many=True)

class MeasurementSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Measurement
        load_instance = True
    station = fields.Nested(StationSchema(only=['name', 'id', 'code', 'longitude', 'latitude']))

measurement_schema = MeasurementSchema()
measurements_schema = MeasurementSchema(many=True)

class PredictionSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Prediction
        load_instance = True
    station = fields.Nested(StationSchema(only=['name', 'id', 'code', 'longitude', 'latitude', 'use_for_maps']))

prediction_schema = PredictionSchema()
predictions_schema = PredictionSchema(many=True)

class RunsSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Runs
        load_instance = True
    
run_schema = RunsSchema()
runs_schema = RunsSchema(many=True)

class HoldoutSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Holdout
        load_instance = True

    station = fields.Nested(StationSchema(only=['id', 'code', 'latitude', 'longitude']))
    measurement = fields.Nested(MeasurementSchema(only=['id', 'time', 'fof2','hmf2','mufd']))

holdout_schema = HoldoutSchema()
holdouts_schema = HoldoutSchema(many=True)

class HoldoutEvalSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = HoldoutEval
        load_instance = True

    holdout = fields.Nested(HoldoutSchema)

holdout_evals_schema = HoldoutEvalSchema(many=True)

class PredEvalSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = PredEval
        load_instance = True

    holdout = fields.Nested(HoldoutSchema)
    measurement = fields.Nested(MeasurementSchema(only=['id', 'time', 'fof2','hmf2','mufd']))

pred_eval_schema = PredEvalSchema()
pred_evals_schema = PredEvalSchema(many=True)

class CosmicEvalSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = CosmicEval
        load_instance = True

cosmic_eval_schema = CosmicEvalSchema()
cosmic_evals_schema = CosmicEvalSchema(many=True)

#You can now use your schema to dump and load your ORM objects.

#Returns latest measurements for all stations in JSON
@app.route("/stations.json" , methods=['GET'])
def stationsjson():
    maxage = request.args.get('maxage', None)
    source = request.args.get('source', None)

    cachekey = 'api;stations.json;' + ('<none>' if maxage is None else maxage) + ';' + ('<none>' if source is None else source)
    ret = memcache.get(cachekey)

    if ret is None:
        sql = "select m1.* from measurement m1 inner join (select station_id, max(time) as maxtime from measurement group by station_id) m2 on m1.station_id=m2.station_id and m1.time=m2.maxtime where 1=1"
        bp = {}

        if maxage is not None:
            sql = sql + " and m1.time >= now() - :maxage * interval '1 second'"
            bp['maxage'] = int(maxage)

        if source is not None:
            sql = sql + " and m1.source = :source"
            bp['source'] = source

        sql = sql + " order by station_id asc"

        qry = db.session.query(Measurement).from_statement(text(sql))
        if len(bp) > 0:
            qry = qry.params(**bp)

        db.session.close()
        
        ret = measurements_schema.dumps(qry)
        memcache.set(cachekey, ret, 60)

    return Response(ret, mimetype='application/json')

@app.route("/pred.json" , methods=['GET'])
def predjson():
    run_id = request.args.get('run_id', None)
    ts = request.args.get('ts', None)

    if run_id is None or ts is None:
        latest = get_latest_run(request.values.get('experiment', None))
        run_id = latest['run_id']
        ahead = int(request.values.get('hours_ahead', 0))
        ts = latest['maps'][ahead]['ts']

    ts = dt.datetime.fromtimestamp(float(ts))

    qry = db.session.query(Prediction).from_statement(
        text("select p.* from prediction p where run_id=:run_id and time=:ts order by station_id asc")).params(
            run_id = run_id,
            ts = ts,
        )
    db.session.close()
    
    ret = predictions_schema.dumps(qry)
    return Response(ret, mimetype='application/json')

@app.route("/pred_sample.json", methods=['GET'])
def pred_sample():
    n_samples = int(request.args.get('samples', 100))

    qry = db.session.query(Prediction).from_statement(
        text("select prediction.* from prediction, (select run_id, min(time) as time from prediction where run_id in (select run_id from prediction order by random() limit :n_samples) group by run_id) sample where prediction.run_id=sample.run_id and prediction.time=sample.time order by run_id asc, station_id asc")).params(
            n_samples = n_samples
    )
    db.session.close()

    ret = predictions_schema.dumps(qry)
    return Response(ret, mimetype='application/json')

@app.route("/pred_series.json", methods=['GET'])
def predseries():
    station_id = request.args.get('station', None)
    experiment = request.args.get('experiment', None)
    run_id = request.args.get('run_id', None)

    cachekey = 'api;pred_series.json;' + ('<none>' if station_id is None else station_id) + ';' + ('<none>' if run_id is None else run_id)
    ret = memcache.get(cachekey)

    if ret is None:
        sql = "select p.* from prediction p"
        if run_id is None:
            sql = sql + " where run_id=(select max(id) from runs where state='finished' and experiment is not distinct from :experiment)"
        else:
            sql = sql + " where run_id=:run_id"

        if station_id is not None:
            sql = sql + " and station_id=:station_id"
        sql = sql + " order by station_id asc, time asc"

        qry = db.session.query(Measurement).from_statement(text(sql))

        if run_id is None:
            qry = qry.params(experiment = experiment)
        else:
            qry = qry.params(run_id = int(run_id))

        if station_id is not None:
            qry = qry.params(station_id = station_id)

        db.session.close()

        result = predictions_schema.dump(qry)

        out = []
        prev_st = None

        for row in result:
            if prev_st is None or row['station']['name'] != prev_st:
                out.append({'station': row['station'], 'pred': []})

            prev_st = row['station']['name']

            out[len(out)-1]['pred'].append({'cs': row['cs'], 'fof2': row['fof2'], 'hmf2': row['hmf2'], 'mufd': row['mufd'], 'time': row['time']})

        ret = json.dumps(out)
        memcache.set(cachekey, ret, 60)

    return Response(ret, mimetype='application/json')

@app.route("/essn.json", methods=['GET'])
def essnjson():
    days = int(request.args.get('days', 7))

    cachekey = 'api;essn.json;' + str(days)
    ret = memcache.get(cachekey)

    if ret is None:
        with db.engine.connect() as conn:
            res = conn.execute(
                text("select extract(epoch from e.time) as time, e.series, e.ssn, e.sfi, e.err from essn e left join runs r on e.run_id=r.id where e.time >= now() - :days * interval '1 day' and r.experiment is null order by time asc").\
                    bindparams(days=days).\
                    columns(time=db.Numeric(asdecimal=False), series=db.Text, ssn=db.Numeric(asdecimal=False), sfi=db.Numeric(asdecimal=False), err=db.Numeric(asdecimal=False))
            )
            rows = res.mappings().all()
            series = {}
            series['24h'] = [ { 'time': round(row['time']), 'ssn': row['ssn'], 'sfi': row['sfi'] } for row in rows if row['series'] == '24h' ]
            series['6h'] = [ { 'time': round(row['time']), 'ssn': row['ssn'], 'sfi': row['sfi'] } for row in rows if row['series'] == '6h' ]

        ret = json.dumps(series)
        memcache.set(cachekey, ret, 60)

    return Response(ret, mimetype='application/json')

@app.route("/irimap.h5", methods=['GET'])
def irimap():
    run_id = request.args.get('run_id', None)
    ts = request.args.get('ts', None)

    if run_id is None or ts is None:
        latest = get_latest_run(request.values.get('experiment', None))
        run_id = latest['run_id']
        ahead = int(request.values.get('hours_ahead', 0))
        ts = latest['maps'][ahead]['ts']

    cachekey = 'api;irimap.h5;%s;%s' % (run_id, ts)
    ret = memcache.get(cachekey)
    
    if ret is None:
        with db.engine.connect() as conn:
            ts = dt.datetime.fromtimestamp(float(ts))
            res = conn.execute(text("select dataset from irimap where run_id=:run_id and time=:ts").\
                               bindparams(run_id=int(run_id), ts=int(ts)).\
                               columns(dataset=db.LargeBinary)
            )

            rows = res.mappings().all()

            if len(rows) == 0:
                return make_response('Not Found', 404)

            ret = rows[0]['dataset']
            memcache.set(cachekey, ret, 3600)

    return Response(ret, mimetype='application/x-hdf5')
        
@app.route("/assimilated.h5", methods=['GET'])
def assimilated():
    run_id = request.args.get('run_id', None)
    ts = request.args.get('ts', None)

    if run_id is None or ts is None:
        latest = get_latest_run(request.values.get('experiment', None))
        run_id = latest['run_id']
        ahead = int(request.values.get('hours_ahead', 0))
        ts = latest['maps'][ahead]['ts']

    cachekey = 'api;assimilated.h5;%s;%s' % (run_id, ts)
    ret = memcache.get(cachekey)

    if ret is None:
        with db.engine.connect() as conn:
            ts = dt.datetime.fromtimestamp(float(ts))
            res = conn.execute(text("select dataset from assimilated where run_id=:run_id and time=:ts").\
                               bindparams(run_id=int(run_id), ts=int(ts)).\
                               columns(dataset=db.LargeBinary)
            )

            rows = res.mappings().all()

            if len(rows) == 0:
                return make_response('Not Found', 404)

            ret = rows[0]['dataset']
            memcache.set(cachekey, ret, 3600)

    return Response(ret, mimetype='application/x-hdf5')
        
@app.route("/ipe.h5", methods=['GET'])
def ipe():
    run_id = request.args.get('run_id', None)
    ts = request.args.get('ts', None)

    if run_id is None or ts is None:
        latest = get_latest_run(request.values.get('experiment', None))
        run_id = latest['run_id']
        ahead = int(request.values.get('hours_ahead', 0))
        ts = latest['maps'][ahead]['ts']

    cachekey = 'api;ipe.h5;%s;%s' % (run_id, ts)
    ret = memcache.get(cachekey)

    if ret is None:
        with db.engine.connect() as conn:
            ts = dt.datetime.fromtimestamp(float(ts))
            res = conn.execute(text("select dataset from ipemap where run_id=:run_id and time=:ts").\
                               bindparams(run_id=int(run_id), ts=int(ts)).\
                               columns(dataset=db.LargeBinary)
            )

            rows = res.mappings().all()

            if len(rows) == 0:
                return make_response('Not Found', 404)

            ret = rows[0]['dataset']
            memcache.set(cachekey, ret, 3600)

    return Response(ret, mimetype='application/x-hdf5')
        
def get_latest_run(experiment=None):
    cachekey = 'api;get_latest_run;%s' % ('<none>' if experiment is None else experiment)
    ret = memcache.get(cachekey)
    if ret is not None:
        return json.loads(ret)

    with db.engine.connect() as conn:
        res = conn.execute(text("select id, run_id, extract(epoch from time) as ts from assimilated where run_id=(select max(id) from runs where state='finished' and experiment is not distinct from :experiment) order by ts asc").\
                           bindparams(experiment=experiment).\
                           columns(id=db.Numeric(asdecimal=False), run_id=db.Numeric(asdecimal=False), ts=db.Numeric(asdecimal=False))
                           )
        rows = res.mappings().all()

        if len(rows) == 0:
            return make_response('Not Found', 404)

        out = {
            'run_id': int(rows[0]['run_id']),
            'maps': [ { 'id': int(x['id']), 'ts': int(x['ts']) } for x in rows ],
        }

        memcache.set(cachekey, json.dumps(out), 60)

        return out

@app.route("/latest_run.json", methods=['GET'])
def latest_run():
    experiment = request.args.get('experiment', None)
    return jsonify(get_latest_run(experiment))

@app.route("/available_maps.json", methods=['GET'])
def available_maps_json():
    past_hours = int(request.args.get('past_hours', '24'))
    future_hours = int(request.args.get('future_hours', '24'))
    experiment = request.args.get('experiment', None)

    with db.engine.connect() as conn:
        sql = """select a1.run_id, a1.ts, a2.start, (case when a1.ts=a2.start then 'now' else ((a1.ts-a2.start+300)/3600)::int::text || 'h' end) as filesuffix 
        from (
            select max(a.run_id) as run_id, extract(epoch from a.time) as ts
            from assimilated a
            join runs r on a.run_id=r.id
            where r.state='finished'
            and r.experiment is not distinct from :experiment
            and a.time >= now() - (:past_hours * interval '1 hour')
            and a.time < now() + (:future_hours * interval '1 hour')
            and a.time >= r.started
            group by a.time
        ) a1
        join (
            select a.run_id, extract(epoch from min(a.time)) as start
            from assimilated a
            join runs r on a.run_id=r.id
            where a.time >= r.started
            group by run_id
        ) a2 
        on a1.run_id=a2.run_id 
        order by a1.ts asc"""

        res = conn.execute(text(sql).\
                           bindparams(experiment=experiment, past_hours=past_hours, future_hours=future_hours).\
                           columns(run_id=db.Numeric(asdecimal=False), ts=db.Numeric(asdecimal=False), start=db.Numeric(asdecimal=False))
                           )
        rows = res.mappings().all()

        if len(rows) == 0:
            return make_response('Not Found', 404)

        rows = [ { 'run_id': int(row['run_id']), 'ts': float(row['ts']), 'start': float(row['start']), 'filesuffix': str(row['filesuffix']) } for row in rows if row['filesuffix'] != '0h' ]
        return jsonify(rows)

@app.route("/band_quality.json", methods=['GET'])
def band_quality_json():
    days = int(request.args.get('days', 7))

    cachekey = 'api;band_quality.json;' + str(days)
    ret = memcache.get(cachekey)

    if ret is None:
        with db.engine.connect() as conn:
            res = conn.execute(
                text("select extract(epoch from time) as time, band, quality from band_quality where time >= now() - :days * interval '1 day' order by time asc").\
                    bindparams(days=days).\
                    columns(time=db.Numeric(asdecimal=False), band=db.Text, quality=db.Numeric(asdecimal=False))
            )
            rows = res.mappings().all()
            series = {}
            for row in rows:
                if series.get(row['band']) is None:
                    series[row['band']] = []
                series[row['band']].append( { 'time': round(row['time']), 'quality': row['quality'] })

        ret = json.dumps(series)
        memcache.set(cachekey, ret, 60)

    return Response(ret, mimetype='application/json')

def maidenhead_to_latlon(grid):
    grid = grid.strip()

    m = re.search(r'^([+-]?[0-9.]+?),([+-]?[0-9.]+)$', grid)
    if m:
        lat = float(m.group(1))
        lon = float(m.group(2))
        return lat, lon

    if len(grid) < 4:
        grid += "55"
    if len(grid) < 6:
        grid += "mm"
    if len(grid) < 8:
        grid += "55"

    lat, lon = maidenhead.to_location(grid)

    return lat, lon

@app.route("/moflof.svg", methods=['GET'])
def mof_lof():
    run_id = request.values.get('run_id', None)
    ts = request.values.get('ts', None)
    metric = request.values.get('metric', 'mof_sp')
    grid = request.values.get('grid', 'fn21wa')
    centered = '1' if request.values.get('centered') in ['true', '1'] else '0'
    warc = '1' if request.values.get('warc', '1') in ['true', '1'] else '0'
    res = float(request.values.get('res', '2'))
    ipe = request.values.get('ipe', '0')

    lat, lon = maidenhead_to_latlon(grid)

    if run_id is None or ts is None:
        latest = get_latest_run()
        run_id = latest['run_id']
        ahead = int(request.values.get('hours_ahead', 0))
        ts = latest['maps'][ahead]['ts']
    else:
        run_id = int(run_id)
        ts = int(ts)

    url = "http://localhost:%s/moflof.svg?run_id=%d&ts=%d&metric=%s&lat=%f&lon=%f&centered=%s&warc=%s&res=%f&ipe=%s" % (os.getenv('RENDERER_PORT'), run_id, ts, metric, lat, lon, centered, warc, res, ipe)
    with urllib.request.urlopen(url) as res:
        content = res.read()
        res = make_response(content)
        res.mimetype = 'image/svg+xml'
        return res

@app.route('/ptp.json', methods=['GET'])
def ptp_json():
    path = request.values.get('path', 'both')
    from_grid = request.values.get('from_grid', None)
    to_grid = request.values.get('to_grid', None)
    debug = request.values.get('debug', '0')

    from_lat, from_lon = maidenhead_to_latlon(from_grid)
    to_lat, to_lon = maidenhead_to_latlon(to_grid)

    latest = get_latest_run()
    run_id = latest['run_id']

    url = "http://localhost:%s/ptp.json?run_id=%d&path=%s&debug=%s&from_lat=%f&from_lon=%f&to_lat=%f&to_lon=%f" % (os.getenv('RAYTRACE_PORT'), run_id, path, debug, from_lat, from_lon, to_lat, to_lon)

    for m in latest['maps']:
        url += '&ts=%d' % (m['ts'])

    with urllib.request.urlopen(url) as res:
        content = res.read()
        res = make_response(content)
        res.mimetype = 'application/json'
        return res

@app.route("/holdout", methods=['GET'])
def get_holdout():
    run_id = int(request.values.get('run_id'))

    cachekey = 'api;holdout;' + str(run_id)
    ret = memcache.get(cachekey)

    if ret is None:
        qry = db.session.query(Holdout).filter(Holdout.run_id == run_id)
        ret = holdouts_schema.dumps(qry)
        memcache.set(cachekey, ret, 60)

    return Response(ret, mimetype='application/json')

@app.route("/holdout_measurements", methods=['POST'])
def post_holdout_measurements():
    num_ho = int(request.form.get('num', 1))
    ret = []
    with db.engine.connect() as conn:
        res = conn.execute(
            text("select m.id as id from measurement m join station s on s.id=m.station_id where s.use_for_essn=true and s.use_for_maps=true and m.time > now() - interval '30 minutes' and m.fof2 is not null and m.hmf2 is not null and m.mufd is not null and (m.cs >= 75 or m.cs = -1) and m.station_id<>67 order by random() limit :num").\
                bindparams(num=num_ho).\
                columns(id=db.Numeric(asdecimal=False))
        )
        for row in res:
            ret.append(int(row[0]))

    return jsonify(ret)

@app.route("/holdout", methods=['POST'])
def post_holdout():
    measurements = request.form.getlist('measurements')
    ret = []

    for meas in measurements:
        with db.engine.connect() as conn:
            res = conn.execute(
                text("select m.id as id, m.station_id as station_id, extract(epoch from m.time) as time from measurement m where m.id=:id ").\
                bindparams(id=int(meas)).\
                columns(id=db.Numeric(asdecimal=False), station_id=db.Numeric(asdecimal=False), time=db.Numeric(asdecimal=False))
            )
            for row in res:
                (measurement_id, station_id, meas_time) = row
                res = conn.execute(
                    text("insert into holdout (station_id, measurement_id) values (:station_id, :measurement_id) returning id").\
                        bindparams(station_id=int(station_id), measurement_id=int(measurement_id))
                )
                (inserted_id,) = res.fetchone()
                ret.append({ 'holdout': inserted_id, 'ts': meas_time })

    return jsonify(ret)

@app.route("/holdout_eval", methods=['GET'])
def get_holdout_eval():
    experiment = request.args.get('experiment', 'crossval_feb_2022')
    qry = db.session.query(HoldoutEval).join(Holdout)
    since = request.args.get('since', None)
    if since is not None:
        ts = dateutil.parser.parse(since)
        ts = ts.replace(tzinfo=dt.timezone.utc)
        qry = qry.filter(Measurement.time >= ts)
    qry = qry.join(Runs).filter_by(experiment=experiment)
    dump = holdout_evals_schema.dump(qry)

    modelmap = {
        'iri': '1-IRI',
        'irimap': '2-IRI+eSSN',
        'assimilated': '3-Full',
        'irtam': '4-IRTAM',
    }

    for row in dump:
        row['holdout']['station']['latitude'] = float(row['holdout']['station']['latitude'])
        row['holdout']['station']['longitude'] = float(row['holdout']['station']['longitude'])

    ret = [ {
        'holdout_id': row['holdout']['id'],
        'time': row['holdout']['measurement']['time'],
        'station': row['holdout']['station'],
        'model': modelmap.get(row['model'], 'unk'),
        'delta_fof2': (float('nan') if row['fof2'] is None else row['fof2']) - row['measurement']['fof2'],
        'delta_mufd': (float('nan') if row['mufd'] is None else row['mufd']) - row['measurement']['mufd'],
        'delta_hmf2': (float('nan') if row['hmf2'] is None else row['hmf2']) - row['measurement']['hmf2'],
        'true_fof2': row['holdout']['measurement']['fof2'],
        'true_mufd': row['holdout']['measurement']['mufd'],
        'true_hmf2': row['holdout']['measurement']['hmf2'],
    } for row in dump ]

    return Response(json.dumps(ret), mimetype='application/json')

@app.route("/pred_eval", methods=['GET'])
def get_pred_eval():
    experiment = request.args.get('experiment', 'pred_jun_2022')
    qry = db.session.query(PredEval).join(Holdout).join(Runs).filter_by(experiment=experiment)

    since = request.args.get('since', None)
    if since is not None:
        ts = dateutil.parser.parse(since)
        ts = ts.replace(tzinfo=dt.timezone.utc)
        qry = qry.filter(Measurement.time >= ts)

    dump = pred_evals_schema.dump(qry)

    modelmap = {
        'iri': '1-IRI',
        'irimap': '2-IRI+eSSN',
        'assimilated': '3-Full',
        'irtam': '4-IRTAM',
    }

    for row in dump:
        row['holdout']['station']['latitude'] = float(row['holdout']['station']['latitude'])
        row['holdout']['station']['longitude'] = float(row['holdout']['station']['longitude'])

    ret = [ {
        'pred_eval_id': row['id'],
        'holdout_id': row['holdout']['id'],
        'data_time': row['holdout']['measurement']['time'],
        'time': row['time'],
        'station': row['holdout']['station'],
        'model': modelmap.get(row['model'], 'unk'),
        'hours_ahead': row['hours_ahead'],
        'delta_fof2': (float('nan') if row['fof2'] is None else row['fof2']) - row['measurement']['fof2'],
        'delta_mufd': (float('nan') if row['mufd'] is None else row['mufd']) - row['measurement']['mufd'],
        'delta_hmf2': (float('nan') if row['hmf2'] is None else row['hmf2']) - row['measurement']['hmf2'],
        'true_fof2': row['measurement']['fof2'],
        'true_mufd': row['measurement']['mufd'],
        'true_hmf2': row['measurement']['hmf2'],
    } for row in dump if row['measurement'] is not None ]

    return Response(json.dumps(ret), mimetype='application/json')

@app.route("/cosmic_eval", methods=['GET'])
def get_cosmic_eval():
    experiments = request.args.getlist('experiment')
    sample = float(request.args.get('sample', 100))
    fmt = request.args.get('format', 'json')

    table = CosmicEval
    if sample != 100:
        table = aliased(table, tablesample(table, sample))

    qry = db.session.query(table).join(Runs).filter(Runs.experiment.in_(experiments)).add_columns(Runs.experiment)

    since = request.args.get('since', None)
    if since is not None:
        ts = dateutil.parser.parse(since)
        ts = ts.replace(tzinfo=dt.timezone.utc)
        qry = qry.filter(CosmicEval.time >= ts)

    generation_after = request.args.get('generation_after', None)
    if generation_after is not None:
        qry = qry.filter(CosmicEval.generation > int(generation_after))

    print(str(qry))

    qry = qry.yield_per(5000)
    if fmt == 'json':
        return Response(dump_streaming(qry, cosmic_eval_schema), mimetype='application/json')
    elif fmt == 'arrow':
        return Response(arrow_streaming(qry.statement, qry.session.bind, remove_fields=['run_id']), mimetype='application/vnd.apache.arrow')
    else:
        raise("unknown format")


@app.route("/sonde_export", methods=['GET'])
def sonde_export():
    station = int(request.args.get('station', None))
    since = request.args.get('since', None)
    sample = int(request.args.get('sample', 100))
    fmt = request.args.get('format', 'arrow')

    table = Measurement
    if sample != 100:
        table = aliased(table, tablesample(table, sample))

    qry = db.session.query(table).filter(Measurement.station_id == station)
    if since is not None:
        ts = dateutil.parser.parse(since)
        ts = ts.replace(tzinfo=dt.timezone.utc)
        qry = qry.filter(Measurement.time >= ts)

    qry = qry.yield_per(5000)
    if fmt == 'json':
        return Response(dump_streaming(qry, measurement_schema), mimetype='application/json')
    elif fmt == 'arrow':
        return Response(arrow_streaming(qry.statement, qry.session.bind), mimetype='application/vnd.apache.arrow')
    else:
        raise("unknown format")

@app.route("/", methods=['GET'])
def static_stations():
      index_path = os.path.join(app.static_folder, 'index.html')
      return send_file(index_path)
      #return render_template('stations.html')
                            #tables=[table.to_html(classes='stationTable', escape=False)],
                            #latestmes = latestmes

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0', port=int(os.getenv('API_PORT')), threaded=False, processes=4)
