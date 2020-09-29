import subprocess
import sys
from datetime import datetime, timezone
import dateutil
import io
import os

import psycopg2
import numpy as np
import wmm2020 as wmm
import h5py

from flask import Flask, request, make_response

def gyf_table(tm):
    year_start = tm.replace(month=1, day=1)
    year_end = year_start.replace(year=year_start.year+1)
    yeardec = tm.year + ((tm - year_start).total_seconds()) / float((year_end - year_start).total_seconds())

    lat, lon = np.meshgrid(
        np.linspace(-90, 90, 181),
        np.linspace(-180, 180, 361),
        indexing='ij'
    )

    mag = wmm.wmm(lat, lon, 100, yeardec)
    z = mag.total.values
    gyf = 0.000028 * z

    return gyf

def generate_map(ssn, sfi, tm):
    iri = subprocess.Popen(
        ['/build/irimap', str(tm.year), str(tm.month), str(tm.day), str(tm.hour), str(tm.minute), str(tm.second), str(ssn)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )

    data = {}
    for key in ['fof2', 'mufd', 'hmf2', 'foe']:
        data[key] = np.zeros((181, 361))


    data['gyf'] = gyf_table(tm)

    for line in iri.stdout:
        lat_, lon_, nmf2_, fof2_, md_, mufd_, hmf2_, foe_ = [float(x) for x in line.split()]
        lat = round(lat_ + 90)
        lon = round(lon_ + 180)

        data['fof2'][lat, lon] = fof2_
        data['mufd'][lat, lon] = mufd_
        data['hmf2'][lat, lon] = hmf2_
        data['foe'][lat, lon] = foe_

    bio = io.BytesIO()
    h5 = h5py.File(bio, 'w')

    for key, val in sorted(data.items()):
        h5.create_dataset('maps/' + key, data=val, compression='gzip', scaleoffset=3)

    h5.create_dataset('/ts', data=np.array(tm.timestamp()))
    h5.create_dataset('/essn/ssn', data=np.array(ssn, np.float32))
    h5.create_dataset('/essn/sfi', data=np.array(sfi, np.float32))

    h5.close()

    return bio.getvalue()


if __name__ == '__main__':
    # Make sure data is initialized before first request.
    x = wmm.wmm(0, 0, 100, 2019.0)


    app = Flask(__name__)

    @app.route('/generate', methods=['POST'])
    def generate():
        dsn = "dbname='%s' user='%s' host='%s' password='%s'" % (os.getenv("DB_NAME"), os.getenv("DB_USER"), os.getenv("DB_HOST"), os.getenv("DB_PASSWORD"))
        con = psycopg2.connect(dsn)

        run_id = request.form.get('run_id', -1)
        tgt = request.form.get('target', None)
        series = request.form.get('series', '24h')

        tm = datetime.fromtimestamp(float(tgt), tz=timezone.utc) if tgt is not None else datetime.now(timezone.utc)

        ssn = None
        sfi = None

        with con.cursor() as cur:
            cur.execute('select ssn, sfi from essn where run_id=%s and series=%s order by time desc limit 1', (run_id, series))
            ssn, sfi = cur.fetchone()

        dataset = generate_map(ssn, sfi, tm)

        with con.cursor() as cur:
            cur.execute('insert into irimap (time, run_id, dataset) values (%s, %s, %s) on conflict (run_id, time) do update set dataset=excluded.dataset', (tm, run_id, dataset))
            con.commit()

        con.close()

        return make_response("OK\n")

    app.run(debug=False, host='0.0.0.0', port=int(os.getenv('IRIMAP_PORT')), threaded=False, processes=4)
