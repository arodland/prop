"""Convert the /kass/mids ionosonde archive (SAO-4 + SAOXML-5) to partitioned parquet.

    uv run data/mids_to_parquet.py /kass/mids /kass/forecast/mids-parquet

Writes <out>/year=YYYY/<URSI>.parquet plus <out>/station.parquet and <out>/report.json.
Re-running skips station-years already written, so it resumes after an interrupt.

Measured over the 2.5Gb link to kass: 621 files/s and 0.14 cores across 1.85M files of DB049,
which is the worst case since nearly every sounding there exists as both a SAO and an XML.
SAO-only stations run nearer 1000 files/s. The job is bound by seek latency on kass's array,
not by CPU -- parsing a SAO file costs 39us against a 4000us cold read -- so files are walked
in directory order (shuffling them across stations halves throughput) and the thread pool only
exists to keep several reads in flight. Expect roughly a day for the full corpus.

.ART files (the pre-2000 binary format) are ignored. Date-only day files (CODE_YYYYDDD.SAO)
concatenate a day's soundings and are read record by record; they mostly repeat soundings that
have their own file, and merge_rows drops those repeats.

The path is the authority for a sounding's time. Filenames on day 000 are the rest of December
31 of the same year. Station-years whose dates cannot be right go to <out>-quarantine (QUARANTINE).

Files that fail are listed in <out>/bad_files.txt rather than only counted, along with in-file
times more than a minute from the path. The truncated XML among them (sizes an exact multiple of
4096) can be fetched again; delete the affected year=*/CODE.parquet to have it re-read.
"""
import bisect
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

SOURCE = "noaa-archive"  # distinct from the live 'noaa' feed already in the DB
THREADS = 32  # flat past this; the array is the wall
MISSING = 9999.0  # SAO-4 fill value

# SAO-4 group table, truncated at the last group we read. (values per line, field width).
# Group 0 geophysical constants, 1 description, 2 timestamp, 3 characteristics, 4 analysis flags.
SAO_GROUPS = [(16, 7), (1, 120), (120, 1), (15, 8), (60, 2)]

# Values per line for every SAO-4 group, in index order, from Data::SAO4's @GROUPS. Only needed to
# find where one record ends in the day files that concatenate a whole day's soundings.
SAO_PER_LINE = [
    16, 1, 120, 15, 60, 16,                              # constants .. doppler table
    15, 15, 40, 120, 15, 15, 15, 40, 120, 15,            # F2 O, F1 O
    15, 15, 40, 120, 15,                                 # E O
    15, 40, 120, 15, 15, 40, 120, 15, 15, 40, 120, 15,   # F2 X, F1 X, E X
    40, 40, 40, 10, 10, 10, 6, 120, 10,                  # median amplitudes .. valley
    15, 40, 120, 15, 15, 40, 120, 15,                    # Es O, E auroral
    15, 15, 15, 120, 120, 120,                           # profile, letters, edit flags
    10, 15, 15, 15,                                      # auroral E profile
]

# Positional order of group 3, from the SAO-4 spec (mirrors Data::SAO4's @CHARACTERISTICS).
SAO_CHARS = (
    "foF2 foF1 M(D) MUF(D) fmin foEs fminF fminE foE fxI h'F h'F2 h'E h'Es zmE yE QF QE DownF "
    "DownE DownEs FF FE D fMUF h'(fMUF) delta_foF2 foEp f(h'F) f(h'F2) foF1p zmF2 zmF1 zhalfNm "
    "foF2p fminEs yF2 yF1 TEC scaleF2 B0 B1 D1 foEa h'Ea foP h'P fbEs typeEs"
).split(" ")

# SAOXML-5 <URSI ID=> codes, used only when an element has no Name attribute (they all had one
# in a 540-file sample spanning every SAOXML-emitting station here). ARTIST writes TEC as 71,
# "I", total electron content to a geostationary satellite -- not 70, which the spec defines as
# "I2000", electron content by the Faraday technique. 80/81 are fminF/fminE, not fminE/QF.
XML_URSI = {
    "00": "foF2", "01": "fxF2", "02": "fzF2", "03": "M(D)", "04": "h'F2", "07": "MUF(D)",
    "09": "scaleF2", "10": "foF1", "11": "fxF1", "14": "h'F1", "16": "h'F", "20": "foE",
    "24": "h'E", "30": "foEs", "31": "fxEs", "32": "fbEs", "34": "h'Es", "36": "typeEs",
    "42": "fmin", "51": "fxI", "60": "f(h'F2)", "61": "f(h'F)", "71": "TEC", "80": "fminF",
    "81": "fminE", "83": "yE", "84": "QF", "85": "QE", "86": "FF", "87": "FE", "88": "fMUF",
    "89": "h'(fMUF)", "90": "zmE", "91": "zmF1", "92": "zmF2", "93": "zhalfNm", "94": "yF2",
    "95": "yF1", "D0": "B0", "D1": "B1", "D2": "D1",
}

# SAOXML writes h`F for h'F and "scale F2" for scaleF2, and names the peak heights hmE/hmF2/hmF1
# where SAO-4's positional table calls them zmE/zmF2/zmF1. Fold both spellings onto the SAO name.
XML_ALIAS = {"hmE": "zmE", "hmF2": "zmF2", "hmF1": "zmF1"}


def canonical(name):
    name = name.replace("`", "'").replace(" ", "").strip()
    return XML_ALIAS.get(name, name)

# Output columns: the characteristics BOTH formats can carry, named as the Postgres
# `measurement` table names them where it has one. fxF2/fzF2/fxF1/fxEs/h'F1 are deliberately
# absent -- SAO-4 has no slot for them, so keeping them would mean a column populated only for
# the handful of XML-only stations.
COLS = {
    "foF2": "fof2", "foF1": "fof1", "MUF(D)": "mufd", "M(D)": "md", "foEs": "foes",
    "foE": "foe", "h'F2": "hf2", "h'E": "he", "zmE": "hme", "zmF2": "hmf2", "zmF1": "hmf1",
    "yF2": "yf2", "yF1": "yf1", "TEC": "tec", "scaleF2": "scalef2", "fbEs": "fbes",
    "h'F": "hf", "h'Es": "hes", "typeEs": "types", "fmin": "fmin", "fminE": "fmine",
    "f(h'F2)": "fhf2", "f(h'F)": "fhf", "yE": "ye", "QF": "qf", "QE": "qe", "FF": "ff",
    "FE": "fe", "zhalfNm": "zhalfnm", "B0": "b0", "B1": "b1", "D1": "d1",
}

# These stations send M(D) in the MUF(D) slot. Same list and threshold as loader/app/load.pl.
MD_AS_MUF = {"MM168", "SD266", "KB548", "MG560", "TK356"}

# Station-years whose dates cannot be right, written to <out>-quarantine instead of <out> so that
# nothing reading the main tree picks them up, but nothing is thrown away either.
QUARANTINE = {
    ("NQJ61", "1980"): "runs ARTIST 0599, a 1999 release; 9 of its files say February 2000",
    ("CO764", "1980"): "runs ARTIST 0189, a 1989 release",
    ("AN438", "1980"): "runs ARTIST 0897, a 1997 release",
}

# lat/lon/name are per-station, not per-sounding, but they are carried on every row so that each
# partition is self-describing and station.parquet can be rebuilt from the output alone. They are
# constant within a file, so run-length encoding makes them almost free.
SCHEMA = pa.schema(
    [("code", pa.string()), ("time", pa.timestamp("us")), ("cs", pa.int16()),
     ("source", pa.string()), ("format", pa.string()), ("name", pa.string()),
     ("lat", pa.float64()), ("lon", pa.float64())]
    + [(c, pa.float64()) for c in COLS.values()]
)

CONF_RE = re.compile(r"Confidence: (\d+)%")
NAME_RE = re.compile(r"NAME\s+([^,]+)")
# CODE_YYYYDDDHHMMSS.SAO, plus the variants the archive actually holds: _SAO.XML and .SAO.XML,
# no seconds (FZA0M and three others), a doubled dot, and date-only day files (CODE_YYYYDDD.SAO,
# sometimes with a letter suffix) that concatenate a whole day's soundings.
FNAME_RE = re.compile(
    r"^([A-Z0-9_]{5})_(\d{4})(\d{3})(?:(\d{2})(\d{2})(\d{2})?)?[A-Za-z]*(?:[._]SAO)?\.+(SAO|XML)$")
URSI_RE = re.compile(r"^[A-Z0-9_]{5}$")  # NI63_ pads its code with an underscore
XML_TIME_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})(?: -\d{3})? (\d{2}):(\d{2}):(\d{2})")


# --- SAO-4 ------------------------------------------------------------------

def _sao_fields(lines, pos, n, per_line, width):
    out = []
    for i in range(0, n, per_line):
        line = lines[pos]
        pos += 1
        out += [line[j * width:(j + 1) * width] for j in range(min(per_line, n - i))]
    return out, pos


def _sao_record(lines, pos, whole):
    """Parse the record starting at lines[pos]: ({group: fields} for groups 0..4, next pos).

    With whole=False it stops after group 4, which is all a single-sounding file needs and
    tolerates trailing groups this module has no table for. With whole=True it steps over every
    group, so that the returned pos is where the next record starts.
    """
    # Only as many counts as are used get read. JR055 and a few others write a mangled index from
    # the doppler table onward (a 127-character first line), after the five this needs.
    n_counts = 80 if whole else len(SAO_GROUPS)
    idx = [int(lines[pos + i // 40][(i % 40) * 3:(i % 40) * 3 + 3] or 0) for i in range(n_counts)]
    pos, got = pos + 2, {}
    for g in range(79 if whole else len(SAO_GROUPS)):  # slot 80 holds the version, not a count
        n = idx[g]
        if not n:
            continue
        if g < len(SAO_GROUPS):
            got[g], pos = _sao_fields(lines, pos, n, *SAO_GROUPS[g])
        elif not whole:
            break
        elif g < len(SAO_PER_LINE):
            pos += -(-n // SAO_PER_LINE[g])
        else:
            raise ValueError(f"records in unknown SAO group {g + 1}")
    return got, pos


def parse_sao(text):
    """{group index: [raw fields]} for groups 0..4 of the first record. Later groups are skipped."""
    return _sao_record(text.replace("\r", "").split("\n"), 0, whole=False)[0]


def sao_records(text):
    """Every record of a day file that concatenates a day's soundings, each as parse_sao gives it."""
    lines = text.replace("\r", "").split("\n")
    pos, out = 0, []
    while pos < len(lines):
        if not lines[pos].strip():
            pos += 1
            continue
        got, pos = _sao_record(lines, pos, whole=True)
        out.append(got)
    return out


def sao_confidence(got):
    """Percent, or -1 when the file says nothing. Mirrors Data::SAO4::confidence."""
    if m := CONF_RE.search(" ".join(got.get(1, ()))):
        return int(m.group(1))
    flags = got.get(4)
    if flags and len(flags) > 9:
        # Perl reads the 2-char field as an integer, then splits the *rendered* integer, so
        # " 4" and "04" both give lower="4" and an empty upper. Reproduced to match the DB.
        s = str(int(flags[9] or 0))
        lo, up = s[0], s[1:2]
        if lo > up:
            cl = int(lo)
        else:
            cl = int(up or lo)
        return 125 - 25 * cl
    return -1


def sao_row(got):
    chars = {}
    for i, s in enumerate(got.get(3, ())):
        if i >= len(SAO_CHARS):
            break
        # A Fortran overflow ('********') or two values run together spoils one field, not the
        # sounding: treat it as missing, as the 9999 fill value would be.
        v = _or_none(float, s)
        if v is not None and v != MISSING:
            chars[SAO_CHARS[i]] = v
    return chars, sao_confidence(got)


def sao_time(got):
    """The in-file timestamp, YYYYDDDMMDDHHMMSS packed across the timestamp group."""
    ts = "".join(got[2][2:19])
    return datetime(int(ts[0:4]), int(ts[7:9]), int(ts[9:11]),
                    int(ts[11:13]), int(ts[13:15]), int(ts[15:17]))


def sao_station(got):
    """(name, lat, lon) from the description line and the geophysical constants."""
    desc = (got.get(1) or [""])[0]
    name = m.group(1).strip() if (m := NAME_RE.search(desc)) else None
    const = got.get(0) or []
    # ND51R's longitude overflows F7.3 as '*******'; station.parquet medians over other files.
    lat = _or_none(float, const[2]) if len(const) > 2 else None
    lon = _or_none(float, const[3]) if len(const) > 3 else None
    return name, lat, lon


# --- SAOXML-5 ---------------------------------------------------------------

def parse_xml(raw):
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError:
        # Some files hold latin1 with no encoding declaration, so they are not legal UTF-8.
        root = ElementTree.fromstring(raw.decode("latin1").encode("utf-8"))
    # Usually <SAORecordList><SAORecord>, but PA836 and two others write a bare <SAORecord>.
    rec = root if root.tag == "SAORecord" else root.find("SAORecord")
    if rec is None:
        raise ValueError("no SAORecord")
    return rec


def xml_confidence(rec):
    sysinfo = rec.find("SystemInfo")
    if sysinfo is not None:
        comment = sysinfo.find("Comments")
        if comment is not None and comment.text and (m := CONF_RE.search(comment.text)):
            return int(m.group(1))
        auto = sysinfo.find("AutoScaler")
        flags = auto.get("ArtistFlags") if auto is not None else None
        if flags:
            f = flags.split()
            if len(f) > 9:
                lo, up = f[9][0], f[9][1:2]
                if lo > up:
                    cl = int(lo)
                else:
                    cl = int(up or lo)
                return 125 - 25 * cl
    return -1


def xml_row(rec):
    chars = {}
    clist = rec.find("CharacteristicList")
    for el in (clist if clist is not None else ()):
        if el.tag != "URSI":
            continue
        # The file's own Name is the authority; the ID table is a fallback for files without one.
        name = canonical(el.get("Name")) if el.get("Name") else XML_URSI.get(el.get("ID"))
        if name is None:
            continue
        try:
            chars[name] = float(el.get("Val"))
        except (TypeError, ValueError):
            pass
    return chars, xml_confidence(rec)


def xml_time(rec):
    """StartTimeUTC looks like '2017-10-19 -292 14:25:02.000' -- the -DDD is the day of year."""
    m = XML_TIME_RE.match(rec.get("StartTimeUTC") or "")
    if not m:
        raise ValueError(f"unparseable StartTimeUTC {rec.get('StartTimeUTC')!r}")
    return datetime(*(int(g) for g in m.groups()))


def xml_station(rec):
    def num(a):
        try:
            return float(rec.get(a))
        except (TypeError, ValueError):
            return None
    return rec.get("StationName"), num("GeoLatitude"), num("GeoLongitude")


# --- shared -----------------------------------------------------------------

def clean(chars, code):
    """The loader/app/load.pl quirk fixes, minus the MUF(D) synthesis."""
    if code in MD_AS_MUF and "M(D)" not in chars and chars.get("MUF(D)", 99) <= 4.2:
        chars["M(D)"] = chars.pop("MUF(D)")
    return {COLS[k]: v for k, v in chars.items() if k in COLS and v != 0}


def sao_tod(got):
    """Time of day from the timestamp group, for day-file records whose date comes from the path.

    Read on its own because some records stamp year 0000 beside a perfectly good time.
    """
    ts = "".join(got[2][2:19])
    h, m, s = int(ts[11:13]), int(ts[13:15]), int(ts[15:17])
    if h > 23 or m > 59 or s > 59:
        raise ValueError(f"no time of day in {ts!r}")
    return timedelta(hours=h, minutes=m, seconds=s)


def _or_none(f, arg):
    try:
        return f(arg)
    except (ValueError, IndexError, KeyError):
        return None


def read_one(path, name):
    """(rows, notes) for one file: usually one sounding, a whole day's for date-only day files.

    notes are non-fatal complaints. A row carries _approx=True when its time is only good to
    about a minute; merge_rows uses that to drop repeats of soundings that have their own file.
    """
    m = FNAME_RE.match(name)
    if not m or (m.group(4) is None and not name.endswith(".SAO")):
        return [], [f"unparseable filename: {path}"]
    code, year, doy, hh, mm, ss, _ = m.groups()
    year, doy = int(year), int(doy)
    # Day 000 is the rest of December 31 of the same year: in every such directory the soundings
    # continue where that year's day 365 leaves off, and the files agree to the second.
    day = datetime(year, 12, 31) if doy == 0 else datetime(year, 1, 1) + timedelta(days=doy - 1)
    try:
        raw = path.read_bytes()
    except OSError as e:
        return [], [f"read failed: {path} ({e})"]

    # (chars, cs), station, when, in-file time (None if unreadable), approx
    found, notes = [], []
    # Go by content, not extension: TM308 writes SAO text into files named _SAO.XML.
    fmt = "xml" if raw.lstrip()[:1] == b"<" else "sao"
    try:
        if fmt == "xml":
            rec = parse_xml(raw)
            when = day + timedelta(hours=int(hh), minutes=int(mm), seconds=int(ss or 0))
            found.append((xml_row(rec), xml_station(rec), when, _or_none(xml_time, rec), ss is None))
        elif hh is None:
            unreadable = 0
            gots = sao_records(raw.decode("ascii", "replace"))
            for got in gots:
                tod = _or_none(sao_tod, got)
                if tod is None:
                    unreadable += 1
                    continue
                found.append((sao_row(got), sao_station(got), day + tod, _or_none(sao_time, got), True))
            if unreadable:
                notes.append(f"unreadable record time: {path} ({unreadable} of {len(gots)} records)")
        else:
            got = parse_sao(raw.decode("ascii", "replace"))
            when = day + timedelta(hours=int(hh), minutes=int(mm), seconds=int(ss or 0))
            found.append((sao_row(got), sao_station(got), when, _or_none(sao_time, got), ss is None))
    except Exception as e:  # a corrupt file must not take the station-year down
        return [], [f"parse failed: {path} ({type(e).__name__}: {e})"]

    rows = []
    for (chars, cs), (name_, lat, lon), when, inner, approx in found:
        row = clean(chars, code)
        if not row:
            continue  # nothing scaled; load.pl skips these too
        row.update(code=code, time=when, cs=cs, source=SOURCE, format=fmt,
                   name=name_, lat=lat, lon=lon, _approx=approx)
        rows.append(row)
        # The path is the authority for when a sounding happened -- it is how the archive is
        # organised, and it sidesteps the decade-shifted years in the IONFM-converted files and
        # the months MZ152 mangles. Under a minute is the scheduled start in the filename against
        # the actual start in the file, 5M files' worth of it, so only larger gaps are recorded.
        if inner is not None and abs(inner - when) >= timedelta(minutes=1) and len(notes) < 5:
            notes.append(f"time mismatch: {path} (file says {inner})")
    return rows, notes


def sounding_files(year_dir):
    """Every .SAO/.XML under a station-year, in directory order (locality is worth ~2x)."""
    individual = year_dir / "individual"
    if not individual.is_dir():
        return  # a station-year can be marked .complete with nothing in it
    for doy in sorted(os.scandir(individual), key=lambda e: e.name):
        if not doy.is_dir():
            # CAJ2M/2012 keeps its first two weeks (1,556 files) directly in individual/.
            if doy.name.endswith(".SAO") or doy.name.endswith(".XML"):
                yield Path(doy.path), doy.name
            continue
        # Almost always 'scaled', but at least one station-year files soundings under 'misc'.
        for sub in sorted(os.scandir(doy.path), key=lambda e: e.name):
            if not sub.is_dir():
                continue
            for f in sorted(os.scandir(sub.path), key=lambda e: e.name):
                if f.name.endswith(".SAO") or f.name.endswith(".XML"):
                    yield Path(f.path), f.name


def merge_rows(rows):
    """One row per sounding, sorted by time.

    A sounding filed as both .SAO and .XML keeps the SAO row -- 96% of the archive is SAO-only
    and should stay consistent -- but takes cs from the XML: SAO almost never carries a
    Confidence comment and falls back to the ARTIST flags, which only resolve 25/50/75/100.
    The same sounding filed twice in one format (under scaled/ and misc/, or as a doubled-dot
    copy) keeps the first.

    Rows timed only to about a minute (no seconds in the filename, or out of a day file) mostly
    repeat soundings that also have their own file, with seconds that differ, so they are kept
    only where no other row lies within a minute of them.
    """
    exact, approx = {}, []
    for row in rows:
        if row.pop("_approx"):
            approx.append(row)
            continue
        prev = exact.get(row["time"])
        if prev is None:
            exact[row["time"]] = row
        elif {prev["format"], row["format"]} == {"sao", "xml"}:
            sao, xml = (prev, row) if prev["format"] == "sao" else (row, prev)
            sao["cs"], sao["format"] = xml["cs"], "sao+xml"
            # SAO descriptions often carry no NAME, where SAOXML always has StationName.
            for k in ("name", "lat", "lon"):
                if sao.get(k) is None:
                    sao[k] = xml.get(k)
            exact[row["time"]] = sao
    times = sorted(exact)
    for row in sorted(approx, key=lambda r: r["time"]):
        t = row["time"]
        i = bisect.bisect_left(times, t)
        if any(abs(times[j] - t) < timedelta(minutes=1) for j in (i - 1, i) if 0 <= j < len(times)):
            continue
        bisect.insort(times, t)
        exact[t] = row
    return [exact[t] for t in times]


def convert_station_year(year_dir, out_path, pool):
    """Parse one station-year, dedupe, write a parquet. Returns (n_rows, notes)."""
    files = list(sounding_files(year_dir))
    if not files:
        return 0, []
    results = list(pool.map(lambda a: read_one(*a), files, chunksize=32))
    notes = [n for _, ns in results for n in ns]
    # A few files sit under another year's directory (FZA0M/2015 holds 306 named for 2016,
    # AU930/2012 156 for 2013). Most have a correctly filed copy, so keeping them would put
    # duplicates in the wrong partition. They are listed instead, to be moved and reloaded.
    year, rows = int(year_dir.name), []
    for (path, _), (rs, _) in zip(files, results):
        if any(r["time"].year != year for r in rs):
            notes.append(f"misfiled: {path} (named for {rs[0]['time'].year})")
        else:
            rows += rs
    rows = merge_rows(rows)
    if not rows:
        return 0, notes
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(out_path)  # atomic, so an interrupted run never leaves a half file to resume past
    return len(rows), notes


def write_stations(out: Path):
    """Derive station.parquet from the partitions on disk, so a resumed run is still complete.

    Position is the median of the per-sounding values rounded to 0.01 degrees. A code whose
    position moves is flagged rather than silently averaged -- it usually means the sonde was
    relocated, or that two sites have shared a URSI code over the years.
    """
    con = duckdb.connect()
    rows = con.execute(f"""
        WITH r AS (
            SELECT code, name,
                   round(lat, 2) AS lat, round(lon, 2) AS lon
            FROM read_parquet('{out}/year=*/*.parquet')
            WHERE lat IS NOT NULL AND lon IS NOT NULL)
        SELECT code,
               -- The URSI code is the honest fallback; SAO descriptions often carry no NAME.
               coalesce(max(name), code) AS name,
               median(lat) AS lat, median(lon) AS lon,
               count(DISTINCT lat) > 1 AS lat_varies,
               count(DISTINCT lon) > 1 AS lon_varies,
               count(DISTINCT lat) AS n_lat, count(DISTINCT lon) AS n_lon
        FROM r GROUP BY code ORDER BY code""").to_arrow_table()
    tmp = out / "station.parquet.tmp"
    pq.write_table(rows, tmp, compression="zstd")
    tmp.replace(out / "station.parquet")  # atomic: the tree may already be in use
    return rows.to_pylist()


def main(root: Path, out: Path):
    report = {"rows": 0, "station_years": 0, "skipped": 0, "notes": {}, "examples": {}}
    out.mkdir(parents=True, exist_ok=True)
    quarantine = out.with_name(out.name + "-quarantine")
    with ThreadPoolExecutor(THREADS) as pool:
        for station in sorted(os.scandir(root), key=lambda e: e.name):
            if not station.is_dir() or not URSI_RE.match(station.name):
                continue
            for year in sorted(os.scandir(station.path), key=lambda e: e.name):
                if not year.is_dir() or not year.name.isdigit():
                    continue
                tree = quarantine if (station.name, year.name) in QUARANTINE else out
                dest = tree / f"year={year.name}" / f"{station.name}.parquet"
                if dest.exists():
                    report["skipped"] += 1
                    continue
                n, notes = convert_station_year(Path(year.path), dest, pool)
                report["rows"] += n
                report["station_years"] += 1
                for note in notes:
                    key = note.split(":")[0]
                    report["notes"][key] = report["notes"].get(key, 0) + 1
                    report["examples"].setdefault(key, note)  # one sample of each kind
                if notes:
                    # Named in full and appended as we go, so an interrupted run keeps them.
                    # Most are files truncated in transfer, which can simply be fetched again;
                    # delete the affected year=*/CODE.parquet afterwards to have them re-read.
                    with (out / "bad_files.txt").open("a") as fh:
                        fh.write("".join(n + "\n" for n in notes))
                print(f"{station.name} {year.name} {n:>7} rows"
                      f"{f'  {len(notes)} bad files' if notes else ''}", flush=True)

    stations = write_stations(out)
    report["stations"] = len(stations)
    report["drift"] = [s for s in stations if s["lat_varies"] or s["lon_varies"]]
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n{report['rows']:,} rows over {report['station_years']} station-years "
          f"({report['skipped']} already done), {len(stations)} stations")
    if report["drift"]:
        print(f"position varies for {len(report['drift'])} codes -- see report.json 'drift':",
              ", ".join(s["code"] for s in report["drift"][:10]))
    if report["notes"]:
        print("bad files:", report["notes"])


def _selfcheck():
    """Pure checks first, then files from /kass/mids when it is mounted."""
    assert len(SAO_PER_LINE) == 60  # Data::SAO4 has 60 groups
    for name, groups in [
        ("DB049_2017292142502.SAO", ("DB049", "2017", "292", "14", "25", "02", "SAO")),
        ("DB049_2017292142502.SAO.XML", ("DB049", "2017", "292", "14", "25", "02", "XML")),
        ("DB049_2017292142502_SAO.XML", ("DB049", "2017", "292", "14", "25", "02", "XML")),
        ("NI63__2010020120000.SAO", ("NI63_", "2010", "020", "12", "00", "00", "SAO")),
        ("FZA0M_20010411952.SAO", ("FZA0M", "2001", "041", "19", "52", None, "SAO")),
        ("JR055_2008264085810..SAO", ("JR055", "2008", "264", "08", "58", "10", "SAO")),
        ("JI91J_2007158.SAO", ("JI91J", "2007", "158", None, None, None, "SAO")),
        ("WP937_1998316ab.SAO", ("WP937", "1998", "316", None, None, None, "SAO")),
    ]:
        m = FNAME_RE.match(name)
        assert m and m.groups() == groups, (name, m and m.groups())
    assert not FNAME_RE.match("DB049_201729214250.SAO")  # 12 digits is neither form

    t = datetime(2020, 5, 1, 12, 0, 0)
    def r(fmt, when, approx=False, cs=50, name=None):
        return dict(time=when, format=fmt, cs=cs, name=name, lat=1.0, lon=2.0, _approx=approx)
    out = merge_rows([r("sao", t, cs=75), r("xml", t, cs=80, name="X")])
    assert len(out) == 1 and out[0]["format"] == "sao+xml" and out[0]["cs"] == 80 and out[0]["name"] == "X"
    out = merge_rows([r("sao", t, cs=75), r("sao", t, cs=25)])  # one sounding filed twice
    assert len(out) == 1 and out[0]["format"] == "sao" and out[0]["cs"] == 75
    out = merge_rows([r("sao", t + timedelta(seconds=3)), r("sao", t, approx=True),
                      r("sao", t + timedelta(minutes=15), approx=True)])
    assert [o["time"] for o in out] == [t + timedelta(seconds=3), t + timedelta(minutes=15)]
    out = merge_rows([r("sao", t, approx=True), r("sao", t + timedelta(seconds=20), approx=True)])
    assert len(out) == 1  # a day file and its lettered twin repeat each other too

    p = Path("/kass/mids/DB049/2017/individual/292/scaled/DB049_2017292142502")
    if not p.with_suffix(".SAO").exists():
        print("selfcheck ok (file checks skipped: /kass/mids not mounted)")
        return
    (sao,), sao_notes = read_one(p.with_suffix(".SAO"), "DB049_2017292142502.SAO")
    (xml,), xml_notes = read_one(Path(str(p) + "_SAO.XML"), "DB049_2017292142502_SAO.XML")
    assert not sao_notes and not xml_notes, (sao_notes, xml_notes)  # path vs in-file time
    assert sao["time"] == datetime(2017, 10, 19, 14, 25, 2), sao["time"]
    assert sao["time"] == xml["time"]
    assert abs(sao["fof2"] - 6.825) < 1e-9, sao["fof2"]
    assert abs(sao["mufd"] - 23.988) < 1e-9, sao["mufd"]
    # Characteristics must agree exactly. cs is expected to differ: SAO quantizes it out of the
    # ARTIST flags (75 here) where XML carries the scaler's real percentage (80).
    for k in set(sao) & set(xml) - {"format", "cs", "_approx"}:
        assert sao[k] == xml[k], f"{k}: sao {sao[k]} != xml {xml[k]}"
    assert sao["cs"] == 75 and xml["cs"] == 80, (sao["cs"], xml["cs"])
    # Group 4 flag "55" -> cl 5 -> 125-125 = 0, matching Data::SAO4 on the same file.
    q = Path("/kass/mids/CO764/2018/individual/016/scaled/CO764_2018016000005.SAO")
    if q.exists():
        got = parse_sao(q.read_bytes().decode("ascii", "replace"))
        assert sao_confidence(got) == 0, sao_confidence(got)
        assert "foF2" not in sao_row(got)[0]  # genuinely blank sounding, not a parse miss
    assert clean({"MUF(D)": 3.5}, "MM168") == {"md": 3.5}  # the mislabel fix
    assert clean({"MUF(D)": 3.5}, "DB049") == {"mufd": 3.5}
    assert clean({"MUF(D)": 9.0}, "MM168") == {"mufd": 9.0}  # above the 4.2 threshold: genuine
    assert clean({"foF2": 0.0, "foE": 2.0}, "DB049") == {"foe": 2.0}  # zeros dropped
    assert canonical("h`F") == "h'F" and canonical("scale F2") == "scaleF2"
    assert canonical("hmF2") == "zmF2"  # XML's name for SAO's zmF2

    # Day 000 is the rest of December 31 of the same year, not of the year before.
    d = Path("/kass/mids/TO536/2010/individual/000/scaled/TO536_2010000151500.SAO")
    (row,), notes = read_one(d, d.name)
    assert row["time"] == datetime(2010, 12, 31, 15, 15) and not notes, (row["time"], notes)

    # A day file holds a whole day: 95 records, every one dated from the path.
    d = Path("/kass/mids/JI91J/2007/individual/158/scaled/JI91J_2007158.SAO")
    rows, notes = read_one(d, d.name)
    assert len(sao_records(d.read_bytes().decode("ascii", "replace"))) == 95
    assert rows and all(x["_approx"] and x["time"].date() == datetime(2007, 6, 7).date() for x in rows)

    # Records stamped year 0000 keep their time of day instead of sinking the file. (This day is
    # nearly all blank soundings; its 97 records match the station's individual files exactly.)
    d = Path("/kass/mids/SN437/2001/individual/025/scaled/SN437_2001025.SAO")
    recs = sao_records(d.read_bytes().decode("ascii", "replace"))
    zero = [g for g in recs if "".join(g[2][2:6]) == "0000"]
    assert len(recs) == 97 and len(zero) == 3 and all(_or_none(sao_tod, g) for g in zero)
    assert not read_one(d, d.name)[1]

    # Files the first full run dropped as 'parse failed' that hold a usable sounding:
    for f, fmt in [
        ("PA836/2010/individual/188/scaled/PA836_2010188231500_SAO.XML", "xml"),  # bare <SAORecord>
        ("TM308/2009/individual/034/scaled/TM308_2009034122050_SAO.XML", "sao"),  # SAO under an XML name
        ("ND51R/2010/individual/310/scaled/ND51R_2010310050000.SAO", "sao"),      # '*******' field
        ("PA836/1998/individual/127/scaled/PA836_1998127230005.SAO", "sao"),      # in-file hour out of range
        ("PQ052/2022/individual/093/scaled/PQ052_2022093154500.SAO", "sao"),      # index mangled after group 5
    ]:
        d = Path("/kass/mids") / f
        rows, notes = read_one(d, d.name)
        assert not any(n.startswith("parse failed") for n in notes), (f, notes)
        assert all(x["format"] == fmt for x in rows), (f, [x["format"] for x in rows])
    print("selfcheck ok")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "selfcheck":
        _selfcheck()
    else:
        main(Path(sys.argv[1]), Path(sys.argv[2]))
