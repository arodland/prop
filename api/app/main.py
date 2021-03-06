from flask import Flask, request, jsonify, render_template, send_file, make_response, Response
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
    station = fields.Nested(StationSchema(only=['name', 'id', 'code', 'longitude', 'latitude']))

prediction_schema = PredictionSchema()
predictions_schema = PredictionSchema(many=True)

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

@app.route("/", methods=['GET'])
def static_stations():
      index_path = os.path.join(app.static_folder, 'index.html')
      return send_file(index_path)
      #return render_template('stations.html')
                            #tables=[table.to_html(classes='stationTable', escape=False)],
                            #latestmes = latestmes

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0', port=int(os.getenv('API_PORT')), threaded=False, processes=4)
