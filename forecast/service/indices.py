"""Daily index population (PLAN.md Phase 5): GFZ Kp/ap/F10.7 and SILSO monthly SSN into Postgres.

    python service/indices.py            # env DB_* as the other services; idempotent upserts
Tables (created if missing):
  indices_daily (date PK, f107_obs, f107_adj, ap, sn, kp3h float[8], ap3h float[8], definitive)
  ssn_monthly   (year, month PK, ssn, definitive)
The forecast service reads indices_daily for its trailing-81-day F10.7 driver and the global token.
Everything here is the observed series; as-of-T estimators truncate at T, never edit these rows.
"""
import io
import os
import urllib.request

import psycopg

SILSO = "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv"
GFZ = "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read().decode()


def parse_gfz(text):
    rows = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        val = lambda x: None if float(x) < 0 else float(x)  # noqa: E731  (-1 marks missing)
        rows.append((f"{f[0]}-{f[1]}-{f[2]}", val(f[25]), val(f[26]), val(f[23]), val(f[24]),
                     [val(x) for x in f[7:15]], [val(x) for x in f[15:23]], f[27] == "1"))
    return rows


def parse_silso(text):
    rows = []
    for line in text.splitlines():
        p = [x.strip() for x in line.split(";")]
        if len(p) < 7:
            continue
        rows.append((int(p[0]), int(p[1]), float(p[3]), p[6] == "1"))
    return rows


def main():
    dsn = "dbname='%s' user='%s' host='%s' port='%s' password='%s'" % (os.getenv("DB_NAME"), os.getenv("DB_USER"), os.getenv("DB_HOST"), os.getenv("PGPORT", "5432"), os.getenv("DB_PASSWORD"))
    gfz, silso = parse_gfz(fetch(GFZ)), parse_silso(fetch(SILSO))
    with psycopg.connect(dsn) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS indices_daily (date date PRIMARY KEY, f107_obs float, f107_adj float, ap float, sn float,
                       kp3h float[], ap3h float[], definitive bool, updated_at timestamptz DEFAULT now());
                       CREATE TABLE IF NOT EXISTS ssn_monthly (year int, month int, ssn float, definitive bool, updated_at timestamptz DEFAULT now(), PRIMARY KEY (year, month))""")
        # only the trailing ~120 days can change (provisional -> definitive); older rows are stable, but upserting all is cheap (35k rows)
        con.cursor().executemany("""INSERT INTO indices_daily (date, f107_obs, f107_adj, ap, sn, kp3h, ap3h, definitive) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (date) DO UPDATE SET f107_obs=excluded.f107_obs, f107_adj=excluded.f107_adj, ap=excluded.ap, sn=excluded.sn,
            kp3h=excluded.kp3h, ap3h=excluded.ap3h, definitive=excluded.definitive, updated_at=now()""", gfz)
        con.cursor().executemany("""INSERT INTO ssn_monthly (year, month, ssn, definitive) VALUES (%s,%s,%s,%s)
            ON CONFLICT (year, month) DO UPDATE SET ssn=excluded.ssn, definitive=excluded.definitive, updated_at=now()""", silso)
        con.commit()
        last = con.execute("SELECT max(date), (SELECT f107_obs FROM indices_daily ORDER BY date DESC LIMIT 1) FROM indices_daily").fetchone()
        print(f"indices_daily: {len(gfz)} rows upserted, latest {last[0]} f107={last[1]}; ssn_monthly: {len(silso)} rows")


if __name__ == "__main__":
    main()
