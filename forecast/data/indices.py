"""Fetch solar/geomagnetic indices to <snapshot>/indices.parquet.

    uv run data/indices.py /kass/forecast/snapshot/2026-09-02/parquet

Sources: SILSO monthly mean sunspot number (V2.0); GFZ daily Kp/ap/Ap/SN/F10.7 since 1932.
Everything here is the *observed* series. As-of-T estimators (PLAN.md, "Indices as-of-T") are
built on top of these by truncating at T, never by editing this file.
"""
import sys
import urllib.request
from pathlib import Path

import duckdb

SILSO = "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv"
GFZ = "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"


def main(out: Path):
    raw = out / "indices_raw"
    raw.mkdir(exist_ok=True)
    for url in (SILSO, GFZ):
        urllib.request.urlretrieve(url, raw / url.rsplit("/", 1)[1])
    con = duckdb.connect()
    # SILSO: year;month;decimal_year;ssn;stdev;n_obs;definitive(1)/provisional(0)
    con.execute(f"""
        CREATE TABLE ssn_monthly AS
        SELECT column0::INT AS year, column1::INT AS month, column3 AS ssn, column6 = 1 AS definitive
        FROM read_csv('{raw}/SN_m_tot_V2.0.csv', delim=';', header=false)""")
    # GFZ: '#' comments, then whitespace-separated:
    # YYYY MM DD days days_m Bsr dB Kp1..Kp8 ap1..ap8 Ap SN F10.7obs F10.7adj D   (D=1 definitive)
    rows = []
    for line in (raw / "Kp_ap_Ap_SN_F107_since_1932.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        val = lambda x: None if float(x) < 0 else float(x)  # -1 marks missing
        rows.append((
            f"{f[0]}-{f[1]}-{f[2]}", [val(x) for x in f[7:15]], [val(x) for x in f[15:23]],
            val(f[23]), val(f[24]), val(f[25]), val(f[26]), f[27] == "1",
        ))
    con.execute("""
        CREATE TABLE daily (date DATE, kp3h DOUBLE[], ap3h DOUBLE[], Ap DOUBLE, SN DOUBLE,
                            f107_obs DOUBLE, f107_adj DOUBLE, definitive BOOLEAN)""")
    con.executemany("INSERT INTO daily VALUES (?::DATE, ?, ?, ?, ?, ?, ?, ?)", rows)
    con.execute(f"COPY ssn_monthly TO '{out}/ssn_monthly.parquet' (FORMAT parquet)")
    con.execute(f"COPY daily TO '{out}/indices_daily.parquet' (FORMAT parquet)")
    for t in ("ssn_monthly", "daily"):
        print(t, con.execute(f"SELECT count(*) FROM {t}").fetchone()[0], "rows")
    print(con.execute("SELECT * FROM daily ORDER BY date DESC LIMIT 2").fetchall())
    print(con.execute("SELECT * FROM ssn_monthly ORDER BY year DESC, month DESC LIMIT 2").fetchall())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
