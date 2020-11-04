from flask import Flask, request, jsonify, render_template, send_file, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from marshmallow import Schema, fields
import json
import os
import datetime as dt
from sqlalchemy import and_, text

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://%s:%s@%s:5432/%s' % (os.getenv("DB_USER"),os.getenv("DB_PASSWORD"),os.getenv("DB_HOST"),os.getenv("DB_NAME"))

# Order matters: Initialize SQLAlchemy before Marshmallow
db = SQLAlchemy(app)
ma = Marshmallow(app)

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

    return jsonify(result.data)

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

@app.route("/pred_series.json", methods=['GET'])
def predseries():
    station_id = request.args.get('station', None)
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

    return jsonify(out)

@app.route("/essn.json", methods=['GET'])
def essnjson():
    days = request.args.get('days', 7)

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

        return jsonify(series)

@app.route("/irimap.h5", methods=['GET'])
def irimap():
    run_id = request.args.get('run_id', None)
    ts = request.args.get('ts', None)

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

        return make_response(rows[0]['dataset'].tobytes(), { 'Content-Type': 'application/x-hdf5' })
        
@app.route("/assimilated.h5", methods=['GET'])
def assimilated():
    run_id = request.args.get('run_id', None)
    ts = request.args.get('ts', None)

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

        return make_response(rows[0]['dataset'].tobytes(), { 'Content-Type': 'application/x-hdf5' })
        
@app.route("/latest_run.json", methods=['GET'])
def latest_run():
    with db.engine.connect() as conn:
        res = conn.execute("select id, run_id, extract(epoch from time) as ts from assimilated where run_id=(select max(id) from runs where state='finished')")
        rows = list(res.fetchall())

        if len(rows) == 0:
            return make_response('Not Found', 404)

        out = {
            'run_id': rows[0]['run_id'],
            'maps': [ { 'id': x['id'], 'ts': x['ts'] } for x in rows ],
        }

        return jsonify(out)

@app.route("/", methods=['GET'])
def static_stations():
      index_path = os.path.join(app.static_folder, 'index.html')
      return send_file(index_path)
      #return render_template('stations.html')
                            #tables=[table.to_html(classes='stationTable', escape=False)],
                            #latestmes = latestmes

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0', port=int(os.getenv('API_PORT')), threaded=False, processes=4)
