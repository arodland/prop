"""Seed a throwaway Postgres with a slice of the snapshot so service/app.py can be exercised end to end.

    uv run service/test_seed.py 2025-06-15T12:00   (env DB_HOST/DB_NAME/DB_USER/DB_PASSWORD/PGPORT)
Creates station, measurement, runs, holdout, essn, assimilated, indices_daily; loads 48 h of measurements before T,
one run row with target_time T (id 1), the essn rows for that window. Prints the run id.
"""
import datetime as dt
import os
import sys

import duckdb
import psycopg

SNAP = "/kass/forecast/snapshot/2026-09-02/parquet"
T = dt.datetime.fromisoformat(sys.argv[1])
dsn = "dbname='%s' user='%s' host='%s' port='%s' password='%s'" % (os.getenv("DB_NAME"), os.getenv("DB_USER"), os.getenv("DB_HOST"), os.getenv("PGPORT", "5432"), os.getenv("DB_PASSWORD"))
d = duckdb.connect()
with psycopg.connect(dsn) as con:
    con.execute("""DROP TABLE IF EXISTS station, measurement, runs, holdout, essn, assimilated, indices_daily;
        CREATE TABLE station (id int PRIMARY KEY, name text, code text, latitude text, longitude text, use_for_essn bool, use_for_maps bool);
        CREATE TABLE measurement (id bigint PRIMARY KEY, station_id int, time timestamp, cs int, source text, fof2 float, hmf2 float, mufd float, tec float);
        CREATE TABLE runs (id bigint PRIMARY KEY, started timestamp, ended timestamp, state text, experiment text, target_time timestamp, use_essn bool, stale_data bool);
        CREATE TABLE holdout (id bigint PRIMARY KEY, run_id bigint, station_id int, measurement_id bigint);
        CREATE TABLE essn (id bigint PRIMARY KEY, time timestamp, series text, run_id bigint, sfi float, ssn float, err float);
        CREATE TABLE assimilated (time timestamp, run_id bigint, dataset bytea, PRIMARY KEY (run_id, time));
        CREATE TABLE indices_daily (date date PRIMARY KEY, f107_obs float, ap3h float[]);""")
    st = d.execute(f"SELECT id, name, code, lat::VARCHAR, lon::VARCHAR, use_for_essn, use_for_maps FROM '{SNAP}/station.parquet'").fetchall()
    con.cursor().executemany("INSERT INTO station VALUES (%s,%s,%s,%s,%s,%s,%s)", st)
    m = d.execute(f"SELECT id, station_id, time, cs, source, fof2, hmf2, mufd, tec FROM '{SNAP}/ionosonde.parquet' WHERE time > ? AND time <= ?", [T - dt.timedelta(hours=48), T + dt.timedelta(hours=24)]).fetchall()
    con.cursor().executemany("INSERT INTO measurement VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)", m)
    con.execute("INSERT INTO runs VALUES (1, %s, NULL, 'created', '2026-09-v4', %s, true, false)", (T, T))
    e = d.execute(f"SELECT id, time, series, run_id, sfi, ssn, err FROM '{SNAP}/essn.parquet' WHERE time BETWEEN ? AND ?", [T - dt.timedelta(hours=6), T]).fetchall()
    con.cursor().executemany("INSERT INTO essn VALUES (%s,%s,%s,%s,%s,%s,%s)", [(r[0], r[1], r[2], 1, r[4], r[5], r[6]) for r in e])
    idx = d.execute(f"SELECT date, f107_obs, ap3h FROM '{SNAP}/indices_daily.parquet' WHERE date BETWEEN ? AND ?", [T.date() - dt.timedelta(days=100), T.date()]).fetchall()
    con.cursor().executemany("INSERT INTO indices_daily VALUES (%s,%s,%s)", [(r[0], r[1], [float(x) if x is not None else None for x in r[2]]) for r in idx])
    con.commit()
    print("seeded:", len(st), "stations,", len(m), "measurements,", len(e), "essn rows,", len(idx), "index days; run_id=1 target", T)
