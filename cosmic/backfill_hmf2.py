import os
import sys
import io
import datetime
import urllib.request
import tarfile
import subprocess
import numpy as np
import netCDF4
import h5py
import hdf5plugin
import json
from pygeodesy.sphericalTrigonometry import LatLon

LatLon.epsilon = 1e-6

from scipy.interpolate import RectBivariateSpline

import psycopg
import psycopg.rows

def gpstime(t):
    return datetime.datetime(1980,1,6) + datetime.timedelta(seconds=t)

def get_iri(tm, lat, lon):

    iri = subprocess.Popen(
            ['/build/iri2020_driver', str(tm.year), str(tm.month), str(tm.day), str(tm.hour), str(tm.minute), str(tm.second), str(lat), str(lon), '0', '0', '0'],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            )

    _ = iri.stdout.readline()

    cols = iri.stdout.readline().split()
    return {'hmf2': float(cols[1]), 'fof2': float(cols[90])}

def db_h5(con, table, run_id, ts):
    with con.cursor() as cur:
        cur.execute("select dataset from "+table+" where run_id=%s and time=%s", (run_id, ts))
        (ds,) = cur.fetchone()
        bio = io.BytesIO(ds)
        h5 = h5py.File(bio, 'r')
        return h5

def spline(table):
    lat = np.linspace(-90, 90, 181)
    lon = np.linspace(-180, 180, 361)
    return RectBivariateSpline(lat, lon, table)

if len(sys.argv) >= 2:
    date = datetime.date.fromisoformat(sys.argv[1])
    year = date.year
    doy = date.toordinal() - datetime.date(year, 1, 1).toordinal() + 1
else:
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    year = yesterday.year
    doy = yesterday.toordinal() - datetime.date(year, 1, 1).toordinal() + 1

urls = []
urls.append('https://data.cosmic.ucar.edu/gnss-ro/cosmic2/provisional/spaceWeather/level2/%d/%03d/ionPrf_prov1_%d_%03d.tar.gz' % (year, doy, year, doy))
print(urls)

bins = {}

num_read = 0
num_loaded = 0
num_attempted = 0

for url in urls:
    try:
        with urllib.request.urlopen(url) as res:
            print(url)
            tf = tarfile.open('cosmic.tar.gz', mode='r|gz', fileobj=res)
            while True:
                memb = tf.next()
                if memb is None:
                    break
                data = tf.extractfile(memb)
                if data is None:
                    continue
                ds = netCDF4.Dataset('dummy.nc', memory=data.read())
                ts = gpstime(ds.edmaxtime)
                rounded = (ts + datetime.timedelta(seconds=450)).replace(second = 0)
                rounded = rounded - datetime.timedelta(minutes = rounded.minute % 15)
                iri = get_iri(ts, ds.edmaxlat, ds.edmaxlon)

                if bins.get(rounded) is None:
                    bins[rounded] = []

                bins[rounded].append({
                    'ts'  : ts,
                    'lat' : ds.edmaxlat,
                    'lon' : ds.edmaxlon,
                    'fof2': ds.critfreq,
                    'hmf2': ds.edmaxalt,
                    'iri_fof2': iri['fof2'],
                    'iri_hmf2': iri['hmf2'],
                })
                num_read += 1
                if num_read % 100 == 0:
                    print("Read %d records" % num_read)
    except Exception as e:
        print(e)
        continue


if num_read % 100 != 0:
    print("Read %d records" % num_read)

dsn = "dbname='%s' user='%s' host='%s' password='%s'" % (os.getenv("DB_NAME"), os.getenv("DB_USER"), os.getenv("DB_HOST"), os.getenv("DB_PASSWORD"))
con = psycopg.connect(dsn)

for key in sorted(bins.keys()):
    records = sorted(bins[key], key=lambda r: r['ts'])
    print("===", key, "===")

    ins = con.cursor()

    with con.cursor() as cur:
        cur.row_factory = psycopg.rows.dict_row
        cur.execute("select a.time, a.run_id, extract(epoch from a.time-r.target_time)::int/3600 as hours_ahead, r.experiment from assimilated a join runs r on a.run_id=r.id where a.time=%s and extract(minute from a.time-r.target_time)=0 and r.experiment is not null order by hours_ahead desc, experiment asc", (key,))
        for row in cur:
            ts = row['time']
            irimap = db_h5(con, 'irimap', row['run_id'], ts)
            assimilated = db_h5(con, 'assimilated', row['run_id'], ts)
            iri_fof2_spline = spline(irimap['/maps/fof2'])
            iri_hmf2_spline = spline(irimap['/maps/hmf2'])
            assimilated_fof2_spline = spline(assimilated['/maps/fof2'])
            assimilated_hmf2_spline = spline(assimilated['/maps/hmf2'])
            sd = json.loads(assimilated['/stationdata/pred'][()])
            sd = [ {'id': x['station.id'], 'latlon': LatLon(x['station.latitude'], x['station.longitude']) } for x in sd ]

            for record in records:
                fof2_true = record['fof2']
                fof2_iri = record['iri_fof2']
                fof2_iri_essn = float(iri_fof2_spline(record['lat'], record['lon'], grid=False))
                fof2_full = float(assimilated_fof2_spline(record['lat'], record['lon'], grid=False))

                hmf2_true = record['hmf2']
                hmf2_iri = record['iri_hmf2']
                hmf2_iri_essn = float(iri_hmf2_spline(record['lat'], record['lon'], grid=False))
                hmf2_full = float(assimilated_hmf2_spline(record['lat'], record['lon'], grid=False))

                ll = LatLon(record['lat'], record['lon'])

                dist = [ {'id': x['id'], 'dist': ll.distanceTo(x['latlon'], wrap=True) / 1000. } for x in sd ]
                dist = sorted(dist, key=lambda x: x['dist'])
                nearest = dist[0]

                ins.execute("update cosmic_eval set hmf2_true=%s, hmf2_iri=%s, hmf2_irimap=%s, hmf2_full=%s where time=%s and abs(latitude-%s)<0.1 and abs(longitude-%s)<0.1 and run_id=%s and hours_ahead=%s", (hmf2_true, hmf2_iri, hmf2_iri_essn, hmf2_full, record['ts'], record['lat'], record['lon'], row['run_id'], row['hours_ahead']))
                num_loaded += ins.rowcount
                num_attempted += 1
                if num_attempted % 10000 == 0:
                    print("Loaded %d/%d records" % (num_loaded, num_attempted))

    con.commit()

if num_attempted % 10000 != 0:
    print("Loaded %d/%d records" % (num_loaded, num_attempted))
