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

.ART files (the pre-2000 binary format) are ignored.

Files that fail to parse are listed in <out>/bad_files.txt rather than only counted. In the
sample so far they are all truncated mid-transfer (every size an exact multiple of 4096), so
they are worth fetching again; delete the affected year=*/CODE.parquet to have them re-read.
"""
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
FNAME_RE = re.compile(r"^([A-Z0-9]{5})_(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(?:[._]SAO)?\.(SAO|XML)$")
URSI_RE = re.compile(r"^[A-Z0-9]{5}$")
XML_TIME_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})(?: -\d{3})? (\d{2}):(\d{2}):(\d{2})")


# --- SAO-4 ------------------------------------------------------------------

def _sao_fields(lines, pos, n, per_line, width):
    out = []
    for i in range(0, n, per_line):
        line = lines[pos]
        pos += 1
        out += [line[j * width:(j + 1) * width] for j in range(min(per_line, n - i))]
    return out, pos


def parse_sao(text):
    """Return {group index: [raw fields]} for groups 0..4. Later groups are never read."""
    lines = text.replace("\r", "").split("\n")
    idx = [int(lines[i // 40][(i % 40) * 3:(i % 40) * 3 + 3] or 0) for i in range(80)]
    pos, got = 2, {}
    for g in range(len(SAO_GROUPS)):
        if idx[g]:
            got[g], pos = _sao_fields(lines, pos, idx[g], *SAO_GROUPS[g])
    return got


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
        v = float(s)
        if v != MISSING:
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
    lat = float(const[2]) if len(const) > 2 else None
    lon = float(const[3]) if len(const) > 3 else None
    return name, lat, lon


# --- SAOXML-5 ---------------------------------------------------------------

def parse_xml(raw):
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError:
        # Some files hold latin1 with no encoding declaration, so they are not legal UTF-8.
        root = ElementTree.fromstring(raw.decode("latin1").encode("utf-8"))
    rec = root.find("SAORecord")
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


def read_one(path, name):
    """(row dict, note) for one sounding file. note is a non-fatal complaint, or None."""
    m = FNAME_RE.match(name)
    if not m:
        return None, f"unparseable filename: {path}"
    code, year, doy, hh, mm, ss = m.group(1), *(int(g) for g in m.groups()[1:6])
    when = datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss)
    try:
        raw = path.read_bytes()
    except OSError as e:
        return None, f"read failed: {path} ({e})"
    try:
        if name.endswith(".SAO"):
            got = parse_sao(raw.decode("ascii", "replace"))
            chars, cs = sao_row(got)
            station = sao_station(got)
            inner = sao_time(got)
            fmt = "sao"
        else:
            rec = parse_xml(raw)
            chars, cs = xml_row(rec)
            station = xml_station(rec)
            inner = xml_time(rec)
            fmt = "xml"
    except Exception as e:  # a corrupt file must not take the station-year down
        return None, f"parse failed: {path} ({type(e).__name__}: {e})"
    row = clean(chars, code)
    if not row:
        return None, None  # nothing scaled; load.pl skips these too
    name_, lat, lon = station
    row.update(code=code, time=when, cs=cs, source=SOURCE, format=fmt,
               name=name_, lat=lat, lon=lon)
    # The path is the authority for when a sounding happened -- it is how the archive is
    # organised, and it sidesteps the decade-shifted years in the IONFM-converted files that
    # Data::SAO4 has to special-case. Disagreement is still worth counting.
    note = None if inner == when else f"time mismatch: {path} (file says {inner})"
    return row, note


def sounding_files(year_dir):
    """Every .SAO/.XML under a station-year, in directory order (locality is worth ~2x)."""
    individual = year_dir / "individual"
    if not individual.is_dir():
        return  # a station-year can be marked .complete with nothing in it
    for doy in sorted(os.scandir(individual), key=lambda e: e.name):
        if not doy.is_dir():
            continue
        # Almost always 'scaled', but at least one station-year files soundings under 'misc'.
        for sub in sorted(os.scandir(doy.path), key=lambda e: e.name):
            if not sub.is_dir():
                continue
            for f in sorted(os.scandir(sub.path), key=lambda e: e.name):
                if f.name.endswith(".SAO") or f.name.endswith(".XML"):
                    yield Path(f.path), f.name


def convert_station_year(year_dir, out_path, pool):
    """Parse one station-year, dedupe, write a parquet. Returns (n_rows, notes)."""
    files = list(sounding_files(year_dir))
    if not files:
        return 0, []
    results = list(pool.map(lambda a: read_one(*a), files, chunksize=32))
    notes = [n for _, n in results if n]
    by_time = {}
    for row, _ in results:
        if not row:
            continue
        # One sounding can appear as both .SAO and .XML. The characteristics agree, so keep the
        # SAO row -- 96% of the archive is SAO-only and should stay consistent. cs is the
        # exception: SAO almost never carries a Confidence comment and falls back to the ARTIST
        # flags, which only resolve 25/50/75/100, while XML usually has the real percentage.
        prev = by_time.get(row["time"])
        if prev is None:
            by_time[row["time"]] = row
            continue
        sao, xml = (prev, row) if prev["format"] == "sao" else (row, prev)
        sao["cs"], sao["format"] = xml["cs"], "sao+xml"
        # SAO descriptions often carry no NAME, where SAOXML always has StationName.
        for k in ("name", "lat", "lon"):
            if sao.get(k) is None:
                sao[k] = xml.get(k)
        by_time[row["time"]] = sao
    rows = [by_time[t] for t in sorted(by_time)]
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
    pq.write_table(rows, out / "station.parquet", compression="zstd")
    return rows.to_pylist()


def main(root: Path, out: Path):
    report = {"rows": 0, "station_years": 0, "skipped": 0, "notes": {}, "examples": {}}
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(THREADS) as pool:
        for station in sorted(os.scandir(root), key=lambda e: e.name):
            if not station.is_dir() or not URSI_RE.match(station.name):
                continue
            for year in sorted(os.scandir(station.path), key=lambda e: e.name):
                if not year.is_dir() or not year.name.isdigit():
                    continue
                dest = out / f"year={year.name}" / f"{station.name}.parquet"
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
    """Parse one known SAO/XML pair and check they agree with each other and with the Perl."""
    p = Path("/kass/mids/DB049/2017/individual/292/scaled/DB049_2017292142502")
    if not p.with_suffix(".SAO").exists():
        print("selfcheck skipped: /kass/mids not mounted")
        return
    sao, sao_note = read_one(p.with_suffix(".SAO"), "DB049_2017292142502.SAO")
    xml, xml_note = read_one(Path(str(p) + "_SAO.XML"), "DB049_2017292142502_SAO.XML")
    assert sao_note is None and xml_note is None, (sao_note, xml_note)  # path vs in-file time
    assert sao["time"] == datetime(2017, 10, 19, 14, 25, 2), sao["time"]
    assert sao["time"] == xml["time"]
    assert abs(sao["fof2"] - 6.825) < 1e-9, sao["fof2"]
    assert abs(sao["mufd"] - 23.988) < 1e-9, sao["mufd"]
    # Characteristics must agree exactly. cs is expected to differ: SAO quantizes it out of the
    # ARTIST flags (75 here) where XML carries the scaler's real percentage (80).
    for k in set(sao) & set(xml) - {"format", "cs"}:
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
    assert FNAME_RE.match("DB049_2017292142502.SAO.XML"), "the .SAO.XML spelling must parse"
    assert FNAME_RE.match("DB049_2017292142502_SAO.XML")
    assert FNAME_RE.match("DB049_2017292142502.SAO")
    print("selfcheck ok")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "selfcheck":
        _selfcheck()
    else:
        main(Path(sys.argv[1]), Path(sys.argv[2]))
