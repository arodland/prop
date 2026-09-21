"""Solar drivers computable at issue time T from data strictly before T (PLAN.md, "Indices as-of-T").

Both return an F10.7-equivalent scalar, which is what PyIRI takes.
"""
import datetime as dt

import PyIRI.main_library as iri


def f107_trailing(con, t, days=81):
    """Mean observed F10.7 over the `days` days ending the day before T."""
    (v,) = con.execute(
        "SELECT avg(f107_obs) FROM daily WHERE date < ?::DATE AND date >= ?::DATE - INTERVAL (?) DAY",
        [t, t, days],
    ).fetchone()
    return float(v)


def ssn_hat(con, t):
    """13-month-smoothed SSN estimate using months strictly before T: mean of the last 13 monthly means.

    ponytail: hold-the-mean extrapolation; McNish-Lincoln if the driver-error check says it matters.
    """
    t = dt.datetime.fromisoformat(str(t))
    (v,) = con.execute(
        """SELECT avg(ssn) FROM (SELECT ssn FROM ssn_monthly
           WHERE make_date(year, month, 1) < make_date(?, ?, 1) ORDER BY year DESC, month DESC LIMIT 13)""",
        [t.year, t.month],
    ).fetchone()
    return float(v)


def f107_from_ssn_hat(con, t):
    return float(iri.R12_2_F107(ssn_hat(con, t)))


DRIVERS = {"f107_81": f107_trailing, "ssn13": f107_from_ssn_hat,
           "f107_27": lambda con, t: f107_trailing(con, t, 27),
           "f107_365": lambda con, t: f107_trailing(con, t, 365)}


def attach_indices(con, snapshot):
    con.execute(f"CREATE VIEW IF NOT EXISTS daily AS SELECT * FROM '{snapshot}/indices_daily.parquet'")
    con.execute(f"CREATE VIEW IF NOT EXISTS ssn_monthly AS SELECT * FROM '{snapshot}/ssn_monthly.parquet'")


def attach_essn(con, snapshot):
    """Production eSSN fits, production runs only (runs.experiment IS NULL)."""
    con.execute(f"""CREATE VIEW IF NOT EXISTS essn AS
        SELECT e.time, e.series, e.sfi, e.ssn, e.err FROM '{snapshot}/essn.parquet' e
        JOIN '{snapshot}/runs.parquet' r ON r.id = e.run_id WHERE r.experiment IS NULL""")


def f107_for_essn(ssn):
    """PyIRI driver equivalent to running the Fortran `irimap` with this eSSN.

    irimap.F90 passes the eSSN as R12 (and derives IG12/F10.7 from it); PyIRI takes F10.7 and
    derives IG12 internally. PyIRI's R12->F10.7->IG12 composition matches irimap's IG12(R12) to
    <0.1, so R12_2_F107(essn) is the exact driver. The essn table's `sfi` column is NOT: PyIRI's
    F10.7->IG12 relation differs from the formula that produced it, underestimating IG12 by ~10.
    """
    return float(iri.R12_2_F107(ssn))


def essn_sfi(con, t, series="24h"):
    """F10.7-equivalent from the latest production eSSN fit at or before T (None if none within 6 h)."""
    row = con.execute(
        "SELECT ssn FROM essn WHERE series = ? AND time <= ?::TIMESTAMP AND time > ?::TIMESTAMP - INTERVAL 6 HOUR ORDER BY time DESC LIMIT 1",
        [series, t, t],
    ).fetchone()
    return None if row is None else f107_for_essn(float(row[0]))
