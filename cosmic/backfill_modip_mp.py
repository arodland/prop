import os
import sys
import io
import datetime
import numpy as np
import ppigrf
import multiprocessing
import queue

import psycopg
import psycopg.rows

def calc_angles(row):
    east, north, up = ppigrf.igrf(lat=float(row['latitude']), lon=float(row['longitude']), h=1, date=row['time'])
    dip_angle = np.degrees(np.arctan2(-up, np.hypot(east, north)))[0,0]
    modip = np.degrees(np.arctan2(np.radians(dip_angle), np.sqrt(np.cos(np.radians(float(row['latitude']))))))
    return row['id'], dip_angle, modip

def consume_queue(rows, results):
    while True:
        try:
            row = rows.get()
            angles = calc_angles(row)
            results.put(angles)
            rows.task_done()
        except ValueError as e:
            return

NPROC=12

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    rows = multiprocessing.JoinableQueue(NPROC*2)
    results = multiprocessing.Queue()

    processes = []
    for i in range(NPROC):
        process = multiprocessing.Process(target=consume_queue, args=(rows, results))
        processes.append(process)
        process.start()

    dsn = "dbname='%s' user='%s' host='%s' password='%s'" % (os.getenv("DB_NAME"), os.getenv("DB_USER"), os.getenv("DB_HOST"), os.getenv("DB_PASSWORD"))
    con = psycopg.connect(dsn)

    query = "select id, time, latitude, longitude from cosmic_eval where dip_angle is null order by time - hours_ahead * interval '1 hour' asc limit 100000"

    i = 0

    with con.cursor() as cur:
        cur.row_factory = psycopg.rows.dict_row
        cur.execute(query)
        for row in cur:
            rows.put(row)
            while True:
                try:
                   rowid, dip_angle, modip = results.get_nowait()
                except queue.Empty as e:
                    break


                con.execute("update cosmic_eval set dip_angle=%s, modip=%s where id=%s" % (dip_angle, modip, rowid))
                i += 1
                if i % 1000 == 0:
                    print("updated %d rows" % i)
                    con.commit()

        rows.close()
        rows.join()

        while True:
            try:
                rowid, dip_angle, modip = results.get_nowait()
            except queue.Empty as e:
                break


            con.execute("update cosmic_eval set dip_angle=%s, modip=%s where id=%s" % (dip_angle, modip, rowid))
            i += 1
            if i % 1000 == 0:
                print("updated %d rows" % i)
                con.commit()

        print("updated %d rows" % i)
        con.commit()

        for p in processes:
            p.kill()
