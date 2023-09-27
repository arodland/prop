import os
import sys
import io
import time
import datetime
import urllib.request
import subprocess
import numpy as np
import psycopg
import psycopg.rows

irtam = None
def get_irtam(tov, tm, lat, lon):
    global irtam
    if irtam is None:
        irtam = subprocess.Popen(
            ['/build/irtam_driver'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )

    irtam.stdin.write("%f %f %s %s\n" % (
        lat, lon,
        tov.strftime('%Y %m %d %H %M'),
        tm.strftime('%Y %m %d %H %M %S'),
        ))
    irtam.stdin.flush()
    line = irtam.stdout.readline()
    data = [ float(x) for x in line.split() ]
    return data

last_dl_time = None
irtam_failed = {}

def ensure_irtam_coefs(metric, tov):
    global last_dl_time
    global irtam_failed

    filename = '/irtam/IRTAM_%s_COEFFS_%s.ASC' % (metric, tov.strftime('%Y%m%d_%H%M'))
    url = 'https://lgdc.uml.edu/rix/gambit-coeffs?charName=%s&time=%s' % (metric, tov.strftime('%Y-%m-%dT%H:%M'))

    if filename in irtam_failed:
        return False

    if os.path.isfile(filename):
        if os.path.getsize(filename) < 16000:
            print('%s is an invalid file' % filename)
            irtam_failed[filename] = True
            return False
        else:
            return True

    tries = 0
    while tries < 3:
        now = datetime.datetime.now()
        if last_dl_time is not None:
            interval = (now - last_dl_time).total_seconds()
            if interval < 30:
                print('delay %.1f seconds' % (30-interval))
                time.sleep(30 - interval)

        last_dl_time = datetime.datetime.now()
        print('download', filename, 'from', url)
        try:
            urllib.request.urlretrieve(url, filename)
            return True
        except Exception as e:
            tries += 1
            print(e, "try: ", tries)
            if tries >= 3:
                with open(filename, 'w') as fh:
                    fh.write("fetch failed\n")
                irtam_failed[filename] = True
                return False
    return False

start_time = datetime.datetime.now()

dsn = "dbname='%s' user='%s' host='%s' password='%s'" % (os.getenv("DB_NAME"), os.getenv("DB_USER"), os.getenv("DB_HOST"), os.getenv("DB_PASSWORD"))
con = psycopg.connect(dsn)

###
### COSMIC
###

query = "select id, time, hours_ahead, latitude, longitude from cosmic_eval where fof2_irtam is null and irtam_failed is null order by time - hours_ahead * interval '1 hour' asc limit 10000"

rows = []
with con.cursor() as cur:
    cur.row_factory = psycopg.rows.dict_row
    cur.execute(query)
    for row in cur:
        tov = row['time'] - datetime.timedelta(hours=row['hours_ahead'])
        tov = tov + datetime.timedelta(seconds=450)
        tov = tov - datetime.timedelta(seconds=tov.second)
        tov = tov - datetime.timedelta(seconds=60*(tov.minute % 15))
        row['tov'] = tov
        rows.append(row)

print("got %d cosmic_eval rows to work on" % len(rows))

rows = sorted(rows, key=lambda r: r['tov'])
i=0
failed=0

with con.cursor() as cur:
    for row in rows:
        tov = row['tov']
        since = datetime.datetime.now() - tov
        if since.total_seconds() < 3*86400 + 1800:
            print("no more data available")
            break

        running_time = datetime.datetime.now() - start_time
        if running_time.total_seconds() > 300:
            print("runtime exceeded")
            break

        fof2_ok = ensure_irtam_coefs('foF2', row['tov'])
        if fof2_ok:
            hmf2_ok = ensure_irtam_coefs('hmF2', row['tov'])

        if fof2_ok and hmf2_ok:
            fof2, hmf2 = get_irtam(row['tov'], row['time'], row['latitude'], row['longitude'])
            con.execute("update cosmic_eval set fof2_irtam=%s, hmf2_irtam=%s where id=%s", (fof2, hmf2, row['id']))
        else:
            con.execute("update cosmic_eval set irtam_failed=true where id=%s", (row['id'],))
            failed += 1

        i += 1
        if i % 100 == 0:
            print("updated %d rows (failed %d)" % (i, failed))
            con.commit()
    if i % 100 != 0:
        print("updated %d rows (failed %d)" % (i, failed))
    con.commit()

###
### HOLDOUT_EVAL
###

query = "select h1.holdout_id, meas.time, sta.latitude, sta.longitude from holdout_eval h1 left join holdout_eval h2 on h1.holdout_id=h2.holdout_id and h2.model='irtam' join holdout ho on h1.holdout_id=ho.id join measurement meas on ho.measurement_id=meas.id join station sta on meas.station_id=sta.id where h1.model='iri' and h2.model is null order by meas.time asc limit 10000"
rows = []
with con.cursor() as cur:
    cur.row_factory = psycopg.rows.dict_row
    cur.execute(query)
    for row in cur:
        tov = row['time']
#        tov = row['time'] - datetime.timedelta(hours=row['hours_ahead'])
        tov = tov + datetime.timedelta(seconds=450)
        tov = tov - datetime.timedelta(seconds=tov.second)
        tov = tov - datetime.timedelta(seconds=60*(tov.minute % 15))
        row['tov'] = tov
        rows.append(row)

print("got %d holdout_eval rows to work on" % len(rows))

rows = sorted(rows, key=lambda r: r['tov'])
i=0
failed=0

with con.cursor() as cur:
    for row in rows:
        tov = row['tov']
        since = datetime.datetime.now() - tov
        if since.total_seconds() < 3*86400 + 1800:
            print("no more data available")
            break

        running_time = datetime.datetime.now() - start_time
        if running_time.total_seconds() > 450:
            print("runtime exceeded")
            break

        fof2_ok = ensure_irtam_coefs('foF2', row['tov'])
        if fof2_ok:
            hmf2_ok = ensure_irtam_coefs('hmF2', row['tov'])

        if fof2_ok and hmf2_ok:
            fof2, hmf2 = get_irtam(row['tov'], row['time'], row['latitude'], row['longitude'])
            con.execute("insert into holdout_eval (holdout_id, model, fof2, hmf2) values (%s, 'irtam', %s, %s)", (row['holdout_id'], fof2, hmf2))
        else:
            con.execute("insert into holdout_eval (holdout_id, model) values (%s, 'irtam')", (row['holdout_id'],))
            failed += 1

        i += 1
        if i % 100 == 0:
            print("updated %d rows (failed %d)" % (i, failed))
            con.commit()
    if i % 100 != 0:
        print("updated %d rows (failed %d)" % (i, failed))
    con.commit()

###
### PRED_EVAL
###

query = "select p1.holdout_id, p1.hours_ahead, p1.time, p1.hours_ahead, p1.measurement_id, sta.latitude, sta.longitude from pred_eval p1 left join pred_eval p2 on p1.holdout_id=p2.holdout_id and p2.model='irtam' join holdout ho on p1.holdout_id=ho.id join measurement meas on ho.measurement_id=meas.id join station sta on meas.station_id=sta.id where p1.model='iri' and p1.measurement_id is not null and p2.model is null order by p1.id asc limit 10000"
rows = []
with con.cursor() as cur:
    cur.row_factory = psycopg.rows.dict_row
    cur.execute(query)
    for row in cur:
        tov = row['time'] - datetime.timedelta(hours=row['hours_ahead'])
        tov = tov + datetime.timedelta(seconds=450)
        tov = tov - datetime.timedelta(seconds=tov.second)
        tov = tov - datetime.timedelta(seconds=60*(tov.minute % 15))
        row['tov'] = tov
        rows.append(row)

print("got %d pred_eval rows to work on" % len(rows))

rows = sorted(rows, key=lambda r: r['tov'])
i=0
failed=0

with con.cursor() as cur:
    for row in rows:
        tov = row['tov']
        since = datetime.datetime.now() - tov
        if since.total_seconds() < 3*86400 + 1800:
            print("no more data available")
            break

        running_time = datetime.datetime.now() - start_time
        if running_time.total_seconds() > 600:
            print("runtime exceeded")
            break

        fof2_ok = ensure_irtam_coefs('foF2', row['tov'])
        if fof2_ok:
            hmf2_ok = ensure_irtam_coefs('hmF2', row['tov'])

        if fof2_ok and hmf2_ok:
            fof2, hmf2 = get_irtam(row['tov'], row['time'], row['latitude'], row['longitude'])
            con.execute("insert into pred_eval (holdout_id, model, time, hours_ahead, measurement_id, fof2, hmf2) values (%s, 'irtam', %s, %s, %s, %s, %s)", (row['holdout_id'], row['time'], row['hours_ahead'], row['measurement_id'], fof2, hmf2))
        else:
            con.execute("insert into pred_eval (holdout_id, model, time, hours_ahead, measurement_id) values (%s, 'irtam', %s, %s, %s)", (row['holdout_id'], row['time'], row['hours_ahead'], row['measurement_id']))
            failed += 1

        i += 1
        if i % 100 == 0:
            print("updated %d rows (failed %d)" % (i, failed))
            con.commit()
    if i % 100 != 0:
        print("updated %d rows (failed %d)" % (i, failed))
    con.commit()
