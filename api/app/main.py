from flask import Flask, request, jsonify, render_template, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from marshmallow import Schema, fields
import json
import os
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

#Generate marshmallow Schemas from your models using ModelSchema

class StationSchema(ma.ModelSchema):
    class Meta:
        model = Station # Fields to expose

station_schema = StationSchema()
stations_schema = StationSchema(many=True)

class MeasurementSchema(ma.ModelSchema):
    class Meta:
        model = Measurement
    station = fields.Nested('StationSchema', only=['name', 'id', 'code', 'longitude', 'latitude'])

measurement_schema = MeasurementSchema()
measurements_schema = MeasurementSchema(many=True)

#You can now use your schema to dump and load your ORM objects.

#Returns latest measurements for all stations in JSON
@app.route("/stations.json" , methods=['GET'])
def stationsjson():

    qry = db.session.query(Measurement).from_statement(
            "select m1.* from measurement m1 inner join (select station_id, max(time) as maxtime from measurement group by station_id) m2 on m1.station_id=m2.station_id and m1.time=m2.maxtime order by station_id asc")

    db.session.close()
    
    result = measurements_schema.dump(qry)

    return jsonify(result.data)

@app.route("/", methods=['GET'])
def static_stations():
      index_path = os.path.join(app.static_folder, 'index.html')
      return send_file(index_path)
      #return render_template('stations.html')
                            #tables=[table.to_html(classes='stationTable', escape=False)],
                            #latestmes = latestmes

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0', port=80)
