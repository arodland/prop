from flask import Flask, request, jsonify, render_template, send_file, make_response, redirect, Response
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from marshmallow import Schema, fields
import json
import os
import datetime as dt
from sqlalchemy import and_, text
import urllib.request
import maidenhead
from pymemcache.client.base import Client as MemcacheClient
import re

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://%s:%s@%s:5432/%s' % (os.getenv("DB_USER"),os.getenv("DB_PASSWORD"),os.getenv("DB_HOST"),os.getenv("DB_NAME"))

# Order matters: Initialize SQLAlchemy before Marshmallow
db = SQLAlchemy(app)
ma = Marshmallow(app)

memcache = MemcacheClient('127.0.0.1', ignore_exc=True, no_delay=True)

#Declare models 

class Station(db.Model):
    id = db.Column(db.Integer,  primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    code = db.Column(db.String, unique=True)
    longitude = db.Column(db.Text)
    latitude = db.Column(db.Text)
    use_for_essn = db.Column(db.Boolean)
    use_for_maps = db.Column(db.Boolean)
    measurements = db.relationship('Measurement', backref='station')

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
    station_name = db.relationship('Station', foreign_keys=[station_id])
    md = db.Column(db.Text)
    def __repr__(self):
        return '<Measurement %r>' % self.id


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.Integer)
    time = db.Column(db.DateTime)
    cs = db.Column(db.Numeric(asdecimal=False))
    fof2 = db.Column(db.Numeric(asdecimal=False))
    mufd = db.Column(db.Numeric(asdecimal=False))
    hmf2 = db.Column(db.Numeric(asdecimal=False))
    log_stdev = db.Column(db.Numeric(asdecimal=False))
    station_id = db.Column(db.Integer, db.ForeignKey('station.id'))
    station_name = db.relationship('Station', foreign_keys=[station_id])
    def __repr__(self):
        return '<Prediction %r>' % self.id

class Holdout(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.Integer)
    station_id = db.Column(db.Integer, db.ForeignKey('station.id'))
    station = db.relationship('Station', foreign_keys=[station_id])
    measurement_id = db.Column(db.Integer, db.ForeignKey('measurement.id'))
    measurement = db.relationship('Measurement', foreign_keys=[measurement_id])
    def __repr__(self):
        return '<Holdout %r: %r %r %r>' % (self.id, self.run_id, self.station_id, self.measurement_id)

class HoldoutEval(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    holdout_id = db.Column(db.Integer, db.ForeignKey('holdout.id'))
    holdout = db.relationship('Holdout', foreign_keys=[holdout_id])
    model = db.Column(db.Text)
    fof2 = db.Column(db.Numeric(asdecimal=False))
    mufd = db.Column(db.Numeric(asdecimal=False))
    hmf2 = db.Column(db.Numeric(asdecimal=False))

class PredEval(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    holdout_id = db.Column(db.Integer, db.ForeignKey('holdout.id'))
    holdout = db.relationship('Holdout', foreign_keys=[holdout_id])
    model = db.Column(db.Text)
    time = db.Column(db.DateTime)
    hours_ahead = db.Column(db.Integer)
    fof2 = db.Column(db.Numeric(asdecimal=False))
    mufd = db.Column(db.Numeric(asdecimal=False))
    hmf2 = db.Column(db.Numeric(asdecimal=False))
    measurement_id = db.Column(db.Integer, db.ForeignKey('measurement.id'))
    measurement = db.relationship('Measurement', foreign_keys=[measurement_id])

class PredEvalMay(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    holdout_id = db.Column(db.Integer, db.ForeignKey('holdout.id'))
    holdout = db.relationship('Holdout', foreign_keys=[holdout_id])
    model = db.Column(db.Text)
    time = db.Column(db.DateTime)
    hours_ahead = db.Column(db.Integer)
    fof2 = db.Column(db.Numeric(asdecimal=False))
    mufd = db.Column(db.Numeric(asdecimal=False))
    hmf2 = db.Column(db.Numeric(asdecimal=False))
    measurement_id = db.Column(db.Integer, db.ForeignKey('measurement.id'))
    measurement = db.relationship('Measurement', foreign_keys=[measurement_id])

#Generate marshmallow Schemas from your models using ModelSchema

class StationSchema(ma.ModelSchema):
    class Meta:
        model = Station # Fields to expose

station_schema = StationSchema()
stations_schema = StationSchema(many=True)

class MeasurementSchema(ma.ModelSchema):
    class Meta:
        model = Measurement
    station = fields.Nested(StationSchema(only=['name', 'id', 'code', 'longitude', 'latitude']))

measurement_schema = MeasurementSchema()
measurements_schema = MeasurementSchema(many=True)

class PredictionSchema(ma.ModelSchema):
    class Meta:
        model = Prediction
    station = fields.Nested(StationSchema(only=['name', 'id', 'code', 'longitude', 'latitude', 'use_for_maps']))

prediction_schema = PredictionSchema()
predictions_schema = PredictionSchema(many=True)

class HoldoutSchema(ma.ModelSchema):
    class Meta:
        model = Holdout

    station = fields.Nested(StationSchema(only=['id', 'code', 'latitude', 'longitude']))
    measurement = fields.Nested(MeasurementSchema(only=['id', 'time', 'fof2','hmf2','mufd']))

holdout_schema = HoldoutSchema()
holdouts_schema = HoldoutSchema(many=True)

class HoldoutEvalSchema(ma.ModelSchema):
    class Meta:
        model = HoldoutEval

    holdout = fields.Nested(HoldoutSchema)

holdout_evals_schema = HoldoutEvalSchema(many=True)

class PredEvalSchema(ma.ModelSchema):
    class Meta:
        model = PredEval

    holdout = fields.Nested(HoldoutSchema)
    measurement = fields.Nested(MeasurementSchema(only=['id', 'time', 'fof2','hmf2','mufd']))

pred_eval_schema = PredEvalSchema()
pred_evals_schema = PredEvalSchema(many=True)

class PredEvalMaySchema(ma.ModelSchema):
    class Meta:
        model = PredEvalMay

    holdout = fields.Nested(HoldoutSchema)
    measurement = fields.Nested(MeasurementSchema(only=['id', 'time', 'fof2','hmf2','mufd']))

pred_eval_may_schema = PredEvalMaySchema()
pred_eval_mays_schema = PredEvalMaySchema(many=True)

#You can now use your schema to dump and load your ORM objects.

#Returns latest measurements for all stations in JSON
@app.route("/stations.json" , methods=['GET'])
def stationsjson():
    maxage = request.args.get('maxage', None)

    cachekey = 'api;stations.json;' + ('<none>' if maxage is None else maxage)
    ret = memcache.get(cachekey)

    if ret is None:
        if maxage is None:
            qry = db.session.query(Measurement).from_statement(
                "select m1.* from measurement m1 inner join (select station_id, max(time) as maxtime from measurement group by station_id) m2 on m1.station_id=m2.station_id and m1.time=m2.maxtime order by station_id asc")
        else:
            qry = db.session.query(Measurement).from_statement(
                "select m1.* from measurement m1 inner join (select station_id, max(time) as maxtime from measurement group by station_id) m2 on m1.station_id=m2.station_id and m1.time=m2.maxtime where m1.time >= now() - :maxage * interval '1 second' order by station_id asc").params(
                maxage=int(maxage),
            )

        db.session.close()
        
        result = measurements_schema.dump(qry)
        ret = json.dumps(result.data)
        memcache.set(cachekey, ret, 60)

    return Response(ret, mimetype='application/json')

@app.route("/pred.json" , methods=['GET'])
def predjson():
    run_id = request.args.get('run_id', None)
    ts = dt.datetime.fromtimestamp(float(request.args.get('ts', None)))

    qry = db.session.query(Measurement).from_statement(
        "select p.* from prediction p where run_id=:run_id and time=:ts order by station_id asc").params(
            run_id = run_id,
            ts = ts,
        )
    db.session.close()
    
    result = predictions_schema.dump(qry)

    return jsonify(result.data)

@app.route("/pred_sample.json", methods=['GET'])
def pred_sample():
    n_samples = int(request.args.get('samples', 100))

    qry = db.session.query(Prediction).from_statement(
        "select prediction.* from prediction, (select run_id, min(time) as time from prediction where run_id in (select run_id from prediction order by random() limit :n_samples) group by run_id) sample where prediction.run_id=sample.run_id and prediction.time=sample.time order by run_id asc, station_id asc").params(
            n_samples = n_samples
    )
    db.session.close()

    result = predictions_schema.dump(qry)

    return jsonify(result.data)

@app.route("/pred_series.json", methods=['GET'])
def predseries():
    station_id = request.args.get('station', None)

    cachekey = 'api;pred_series.json;' + ('<none>' if station_id is None else station_id)
    ret = memcache.get(cachekey)

    if ret is None:
        sql = "select p.* from prediction p where run_id=(select max(id) from runs where state='finished')"
        if station_id is not None:
            sql = sql + " and station_id=:station_id"
        sql = sql + " order by station_id asc, time asc"

        qry = db.session.query(Measurement).from_statement(sql)
        if station_id is not None:
            qry = qry.params(station_id = station_id)

        db.session.close()

        result = predictions_schema.dump(qry)

        out = []
        prev_st = None

        for row in result.data:
            if prev_st is None or row['station_name'] != prev_st:
                out.append({'station': row['station'], 'pred': []})

            prev_st = row['station_name']

            out[len(out)-1]['pred'].append({'cs': row['cs'], 'fof2': row['fof2'], 'hmf2': row['hmf2'], 'mufd': row['mufd'], 'time': row['time']})

        ret = json.dumps(out)
        memcache.set(cachekey, ret, 60)

    return Response(ret, mimetype='application/json')

@app.route("/essn.json", methods=['GET'])
def essnjson():
    days = request.args.get('days', 7)

    cachekey = 'api;essn.json;' + str(days)
    ret = memcache.get(cachekey)

    if ret is None:
        with db.engine.connect() as conn:
            res = conn.execute(
                text("select extract(epoch from time) as time, series, ssn, sfi, err from essn where time >= now() - :days * interval '1 day' order by time asc").\
                    bindparams(days=days).\
                    columns(time=db.Numeric(asdecimal=False), series=db.Text, ssn=db.Numeric(asdecimal=False), sfi=db.Numeric(asdecimal=False), err=db.Numeric(asdecimal=False))
            )
            rows = list(res.fetchall())
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

    cachekey = 'api;irimap.h5;%s;%s' % (('<none>' if run_id is None else run_id), ('<none>' if ts is None else ts))
    ret = memcache.get(cachekey)
    
    if ret is None:
        with db.engine.connect() as conn:
            if run_id is not None and ts is not None:
                ts = dt.datetime.fromtimestamp(float(ts))
                res = conn.execute("select dataset from irimap where run_id=%s and time=%s",
                    (run_id, ts),
                )
            else:
                res = conn.execute("select dataset from irimap order by run_id asc, time desc limit 1")

            rows = list(res.fetchall())

            if len(rows) == 0:
                return make_response('Not Found', 404)

            ret = rows[0]['dataset'].tobytes()
            memcache.set(cachekey, ret, 3600)

    return Response(ret, mimetype='application/x-hdf5')
        
@app.route("/assimilated.h5", methods=['GET'])
def assimilated():
    run_id = request.args.get('run_id', None)
    ts = request.args.get('ts', None)

    cachekey = 'api;assimilated.h5;%s;%s' % (('<none>' if run_id is None else run_id), ('<none>' if ts is None else ts))
    ret = memcache.get(cachekey)

    if ret is None:
        with db.engine.connect() as conn:
            if run_id is not None and ts is not None:
                ts = dt.datetime.fromtimestamp(float(ts))
                res = conn.execute("select dataset from assimilated where run_id=%s and time=%s",
                    (run_id, ts),
                )
            else:
                res = conn.execute("select dataset from assimilated order by run_id desc, time asc limit 1")

            rows = list(res.fetchall())

            if len(rows) == 0:
                return make_response('Not Found', 404)

            ret = rows[0]['dataset'].tobytes()
            memcache.set(cachekey, ret, 3600)

    return Response(ret, mimetype='application/x-hdf5')
        
def get_latest_run():
    with db.engine.connect() as conn:
        res = conn.execute("select id, run_id, extract(epoch from time) as ts from assimilated where run_id=(select max(id) from runs where state='finished') order by ts asc")
        rows = list(res.fetchall())

        if len(rows) == 0:
            return make_response('Not Found', 404)

        out = {
            'run_id': rows[0]['run_id'],
            'maps': [ { 'id': x['id'], 'ts': x['ts'] } for x in rows ],
        }

        return out

@app.route("/latest_run.json", methods=['GET'])
def latest_run():
    return jsonify(get_latest_run())

@app.route("/available_maps.json", methods=['GET'])
def available_maps_json():
    past_hours = request.args.get('past_hours', '24')
    future_hours = request.args.get('future_hours', '24')

    with db.engine.connect() as conn:
        sql = """select a1.run_id, a1.ts, a2.start, (case when a1.ts=a2.start then 'now' else ((a1.ts-a2.start+300)/3600)::int::text || 'h' end) as filesuffix 
        from (
            select max(a.run_id) as run_id, extract(epoch from a.time) as ts
            from assimilated a
            join runs r on a.run_id=r.id
            where r.state='finished'
            and a.time >= now() - (%s * interval '1 hour')
            and a.time < now() + (%s * interval '1 hour')
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

        res = conn.execute(sql, (past_hours, future_hours))
        rows = list(res.fetchall())

        if len(rows) == 0:
            return make_response('Not Found', 404)

        rows = [ { 'run_id': int(row['run_id']), 'ts': float(row['ts']), 'start': float(row['start']), 'filesuffix': str(row['filesuffix']) } for row in rows if row['filesuffix'] != '0h' ]
        return jsonify(rows)

@app.route("/band_quality.json", methods=['GET'])
def band_quality_json():
    days = request.args.get('days', 7)

    cachekey = 'api;band_quality.json;' + str(days)
    ret = memcache.get(cachekey)

    if ret is None:
        with db.engine.connect() as conn:
            res = conn.execute(
                text("select extract(epoch from time) as time, band, quality from band_quality where time >= now() - :days * interval '1 day' order by time asc").\
                    bindparams(days=days).\
                    columns(time=db.Numeric(asdecimal=False), band=db.Text, quality=db.Numeric(asdecimal=False))
            )
            rows = list(res.fetchall())
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
    res = float(request.values.get('res', '2'))

    lat, lon = maidenhead_to_latlon(grid)

    if run_id is None or ts is None:
        latest = get_latest_run()
        run_id = latest['run_id']
        ahead = int(request.values.get('hours_ahead', 0))
        ts = latest['maps'][ahead]['ts']
    else:
        run_id = int(run_id)
        ts = int(ts)

    url = "http://localhost:%s/moflof.svg?run_id=%d&ts=%d&metric=%s&lat=%f&lon=%f&centered=%s&res=%f" % (os.getenv('RENDERER_PORT'), run_id, ts, metric, lat, lon, centered, res)
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
    qry = db.session.query(Holdout).filter(Holdout.run_id == run_id)
    ho = holdouts_schema.dump(qry)
    return Response(json.dumps(ho.data), mimetype='application/json')

@app.route("/holdout", methods=['POST'])
def post_holdout():
    num_ho = int(request.form.get('num', 1))
    ret = []

    with db.engine.connect() as conn:
        res = conn.execute(
            text("select m.id as id, m.station_id as station_id, extract(epoch from m.time) as time from measurement m join station s on s.id=m.station_id where s.use_for_essn=true and s.use_for_maps=true and m.time > now() - interval '30 minutes' and m.fof2 is not null and m.hmf2 is not null and m.mufd is not null and (m.cs >= 75 or m.cs = -1) and m.station_id<>67 order by random() limit :num").\
                bindparams(num=num_ho).\
                columns(id=db.Numeric(asdecimal=False), station_id=db.Numeric(asdecimal=False), time=db.Numeric(asdecimal=False))
        )
        for row in res:
            (measurement_id, station_id, meas_time) = row
            res = conn.execute(
                text("insert into holdout (station_id, measurement_id) values (:station_id, :measurement_id) returning id").\
                    bindparams(station_id=station_id, measurement_id=measurement_id)
            )
            (inserted_id,) = res.fetchone()
            ret.append({ 'holdout': inserted_id, 'ts': meas_time })

    return jsonify(ret)

@app.route("/holdout_eval", methods=['GET'])
def get_holdout_eval():
    qry = db.session.query(HoldoutEval)
    dump = holdout_evals_schema.dump(qry)

    modelmap = {
        'iri': '1-IRI',
        'irimap': '2-IRI+eSSN',
        'assimilated': '3-Full',
    }

    for row in dump.data:
        row['holdout']['station']['latitude'] = float(row['holdout']['station']['latitude'])
        row['holdout']['station']['longitude'] = float(row['holdout']['station']['longitude'])

    ret = [ {
        'holdout_id': row['holdout']['id'],
        'time': row['holdout']['measurement']['time'],
        'station': row['holdout']['station'],
        'model': modelmap.get(row['model'], 'unk'),
        'delta_fof2': row['fof2'] - row['holdout']['measurement']['fof2'],
        'delta_mufd': row['mufd'] - row['holdout']['measurement']['mufd'],
        'delta_hmf2': row['hmf2'] - row['holdout']['measurement']['hmf2'],
        'true_fof2': row['holdout']['measurement']['fof2'],
        'true_mufd': row['holdout']['measurement']['mufd'],
        'true_hmf2': row['holdout']['measurement']['hmf2'],
    } for row in dump.data ]

    return Response(json.dumps(ret), mimetype='application/json')

@app.route("/pred_eval", methods=['GET'])
def get_pred_eval():
    dataset = request.args.get('dataset', 'new')
    if dataset == 'may':
        qry = db.session.query(PredEvalMay)
        dump = pred_eval_mays_schema.dump(qry)
    else:
        qry = db.session.query(PredEval)
        dump = pred_evals_schema.dump(qry)


    modelmap = {
        'iri': '1-IRI',
        'irimap': '2-IRI+eSSN',
        'assimilated': '3-Full',
    }

    for row in dump.data:
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
        'delta_fof2': row['fof2'] - row['measurement']['fof2'],
        'delta_mufd': row['mufd'] - row['measurement']['mufd'],
        'delta_hmf2': row['hmf2'] - row['measurement']['hmf2'],
        'true_fof2': row['measurement']['fof2'],
        'true_mufd': row['measurement']['mufd'],
        'true_hmf2': row['measurement']['hmf2'],
    } for row in dump.data if row['measurement'] is not None ]

    return Response(json.dumps(ret), mimetype='application/json')

@app.route("/", methods=['GET'])
def static_stations():
      index_path = os.path.join(app.static_folder, 'index.html')
      return send_file(index_path)
      #return render_template('stations.html')
                            #tables=[table.to_html(classes='stationTable', escape=False)],
                            #latestmes = latestmes

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0', port=int(os.getenv('API_PORT')), threaded=False, processes=4)
