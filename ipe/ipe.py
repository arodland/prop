from datetime import datetime, timezone, timedelta
import numpy as np
import os
import io
import re
import math
import h5py
import hdf5plugin
import netCDF4
import tarfile
from scipy.interpolate import RectBivariateSpline

import psycopg

import subprocess
import urllib.request, urllib.error
import sys

from flask import Flask, request, make_response

def get_ncdata(ts):
    ts = ts.replace(second=0)
    ts = ts - timedelta(minutes = ts.minute % 5)
    run_ts = ts - timedelta(hours = ts.hour % 6)
    for i in range(7):
        if i == 6:
            raise "file not found"

        nc_url = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/wfs/prod/wfs.%04d%02d%02d/%02d/wfs.t%02dz.ipe05.%04d%02d%02d_%02d%02d%02d.nc' % (
            run_ts.year, run_ts.month, run_ts.day, run_ts.hour,
            run_ts.hour,
            ts.year, ts.month, ts.day,
            ts.hour, ts.minute, ts.second
        )
        print(nc_url)

        try:
            with urllib.request.urlopen(nc_url) as res:
                content = res.read()

            ds = netCDF4.Dataset('dummy.nc', memory=content)
            return {
                'hmf2': ds['HmF2'][:].data,
                'nmf2': ds['NmF2'][:].data,
                'lat': ds['lat'][:].data,
                'lon': ds['lon'][:].data,
            }

        except urllib.error.HTTPError as e:
            if e.code == 404:
                run_ts = run_ts - timedelta(hours=6)
            else:
                raise e
        else:
            break

def resample(ilat, ilon, ds):
    olat, olon = np.meshgrid(
        np.linspace(-90, 90, 181),
        np.hstack((np.linspace(180, 359, 180), np.linspace(0, 180, 181))),
        indexing='ij',
    )
    spline = RectBivariateSpline(ilat, ilon, ds)
    return spline(olat.flatten(), olon.flatten(), grid=False).reshape(olon.shape)

def get_foe(ts):
    iri = subprocess.Popen(
        ['/build/irimap', str(ts.year), str(ts.month), str(ts.day), str(ts.hour), str(ts.minute), str(ts.second), '-100'],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )
    foe = np.zeros((181, 361))
    for line in iri.stdout:
        lat_, lon_, nmf2_, fof2_, md_, mufd_, hmf2_, foe_ = [float(x) for x in line.split()]
        lat = round(lat_ + 90)
        lon = round(lon_ + 180)
        foe[lat, lon] = foe_

    return foe

app = Flask(__name__)
@app.route('/generate', methods=['POST'])
def generate():
    dsn = "dbname='%s' user='%s' host='%s' password='%s'" % (os.getenv("DB_NAME"), os.getenv("DB_USER"), os.getenv("DB_HOST"), os.getenv("DB_PASSWORD"))
    con = psycopg.connect(dsn)

    run_id = int(request.form.get('run_id', -1))
    tgt = int(request.form.get('target', None))
    ts = datetime.fromtimestamp(float(tgt), tz=timezone.utc)

    data = get_ncdata(ts)
    
    # Repair holes at the south geomagnetic pole (8, 31) and north geomagnetic pole (86, 68-72)
    for ds in (data['nmf2'], data['hmf2']):
        ds[8, 31] = (ds[7, 30] + ds[8, 30] + ds[9, 30] + ds[7, 31] + ds[9, 31] + ds[7, 32] + ds[8, 32] + ds[9, 32]) / 8

        ds[86, 68] = (ds[85, 67] + ds[85, 68] + ds[85, 69] + ds[86, 67] + ds[87, 67] + ds[87, 68] + ds[87, 69]) / 7
        ds[86, 69] = (ds[85, 68] + ds[85, 69] + ds[85, 70] + ds[87, 68] + ds[87, 69] + ds[87, 70]) / 6
        ds[86, 70] = (ds[85, 69] + ds[85, 70] + ds[85, 71] + ds[87, 69] + ds[87, 70] + ds[87, 71]) / 6
        ds[86, 71] = (ds[85, 70] + ds[85, 71] + ds[85, 72] + ds[87, 70] + ds[87, 71] + ds[87, 72]) / 6
        ds[86, 72] = (ds[85, 71] + ds[85, 72] + ds[85, 73] + ds[86, 73] + ds[87, 71] + ds[87, 72] + ds[87, 73]) / 7

    # Replicate lon=0 data at lon=360
    data['lon'] = np.append(data['lon'], 360)
    data['hmf2'] = np.column_stack(( data['hmf2'], data['hmf2'][:,0] ))
    data['nmf2'] = np.column_stack(( data['nmf2'], data['nmf2'][:,0] ))
    # Upscale to 1°x1°
    data['hmf2'] = resample(data['lat'], data['lon'], data['hmf2'])
    data['nmf2'] = resample(data['lat'], data['lon'], data['nmf2'])

    # Convert NmF2 to foF2
    data['fof2'] = np.sqrt(data['nmf2'] / 1.24e10)

    # Shimazaki/Dudeney using IRI-estimated foE
    data['M'] = 1490. / (data['hmf2'] + 176)
    data['foe'] = get_foe(ts)
    data['dM'] = 0.253/np.clip(data['fof2'] / data['foe'] - 1.215, 0.01, None) - 0.012

    data['mufd'] = data['fof2'] * (data['M'] + data['dM'])

    bio = io.BytesIO()
    h5 = h5py.File(bio, 'w')

    h5.create_dataset('/ts', data=np.array(ts.timestamp()))
    for key in ['fof2', 'mufd', 'hmf2', 'foe']:
        h5.create_dataset('maps/' + key, data=data[key], **hdf5plugin.SZ(absolute=0.001))

    h5.close()
    dataset = bio.getvalue()

    with con.cursor() as cur:
        cur.execute('insert into ipemap (time, run_id, dataset) values (%s, %s, %s) on conflict (run_id, time) do update set dataset=excluded.dataset', (ts, run_id, dataset))
        con.commit()

    con.close()
    return make_response("OK\n")

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.getenv('ESSN_PORT')))
