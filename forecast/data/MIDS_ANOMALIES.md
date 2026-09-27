# Known anomalies in the mids archive

Station position and identity problems in `/kass/mids`, as loaded into
`/kass/forecast/mids-parquet` by `mids_to_parquet.py`. The data is deliberately left as the files
report it; this is the record of what is wrong with it. Found in September 2026 against the full
download (60.9M files).

TLDR: none of this is a loader bug. Every bad position is the literal constants line of the file.

- Six codes hold another station's soundings. Positions come from `station.parquet`, which takes
  the median per code, so these rows are placed at the wrong site and most of them duplicate the
  true station. GR13L is the big one: 91.6k rows of College, Alaska filed as Grahamstown.
- Several stations carry corrupted coordinates in their own files. Harmless while positions come
  from `station.parquet`; wrong if anything uses the row-level `lat`/`lon`.
- ND51R has no row in `station.parquet`, so a join on it drops all 459 of its rows.

The parquet tree keeps all of this as reported. `mids_to_duckdb.py` builds the working copy,
`/kass/forecast/mids.duckdb`, without the misfiled rows (its `EXCLUDE` and `REPEATS` lists, with
counts in the `excluded` table) and with M(D)/MUF(D) filled as `loader/app/load.pl` does.

Distances below are from the code's main position (the lat/lon carried by the most rows), with
longitude taken modulo 360. Under 25 km the differences are precision (40.0 against 40.3) or short
moves, and are not listed.

## Another station's soundings under this code

The file's description line names a different station than the directory it sits in, and its
coordinates are that station's too. "Twins" are rows under the true station's code within two
minutes carrying identical foF2, MUF(D) and hmF2.

| code | period | rows | file says | twins | notes |
|---|---|---|---|---|---|
| GR13L | 1999-01-04 to 2003-12-31 | 86,196 at 64.9/212.0 | `DGS-256 082/CO764` | 74,268 with CO764, a few with GA762/GSJ53 | about 13k have no matching copy |
| GR13L | 2002-11-28 to 2003-03-07 | 5,453 at 64.9/102.0 | `082/CO764` | with CO764 | same corrupted longitude as CO764's own files then |
| FF051 | 2009-10-27 to 2009-11-10 | 1,219 at 37.9/284.5 | `DGS-256 081/WP937` | 1,117 with WP937 | Wallops |
| CW46O | 2008-01-04 to 2008-01-05 | 130 at 38.0/23.5 | `DPS-4 204/AT138` | 128 with AT138 | Athens; plus one row at -34.05/150.67 twinned with CN53L |
| OS768 | 2000-10-25 to 2000-10-27 | the whole code | `DGS-256 096/SN437` | 184 with SN437 | one day file; same site, so position is unaffected |
| SL051 | 2000 to 2008 | the whole code, 4,835 files | `DPS-1 052/` (Chilton's DPS) | see below | same position as RL052 |
| AC843 | 2012-02-20 to 2012-09-11 | the whole code, 13,367 rows | `INL43` | none | INL43's own directory holds no data files, so this is INL43's only copy; carries INL's position (43.81/247.32) |

SL051 was Slough, whose IPS42 was replaced by the Chilton DPS (RL052) in December 1993, 50 km
away. Everything under SL051 in this archive is 2000-2008 Chilton DPS data at Chilton's
coordinates. RL052 has a sounding at the same timestamp for 4,715 of SL051's 4,813 rows, with the
same cs. In 2000-01 about 80% of those pairs differ in foF2 by a median 0.05-0.07 MHz (the same
ionogram autoscaled twice); in 2007-08, 84% agree exactly. 98 SL051 rows fill gaps in RL052.

ND51R and ND61R are both Niue. ND51R's 2010 directory holds files declaring ND61R, and 391 of
ND51R's 459 rows (November 2010) are identical to ND61R soundings.

## Right station, corrupted coordinates

The file names its own station; only the constants line is wrong.

| code | period | rows | file has | should be | cause |
|---|---|---|---|---|---|
| BC840 | 1980-02-25 to 2002-04-14 | 37,471 | 40.000 / 465.300 | 40.0 / 254.7 | KEL IPS-71 files: west longitude written as positive, plus 360 |
| BC840 | 1980-02-24 to 1980-02-25 | 51 | 30.216 / 40.000 | 40.0 / 254.7 | junk constants |
| PA836 | 1998-12-05 to 1999-10-21 | 13,743 | 43.8 / 239.5 | 34.8 / 239.5 | digits swapped; WMO ID is KVBG, Vandenberg |
| CO764 | 2002-11-27 to 2003-03-07 | 7,120 | 64.9 / 102.0 | 64.9 / 212.0 | |
| ND61R | 2010-06-24 to 2010-11-25 | 694 | -19.07 / 0.0 | -19.07 / 190.07 | longitude field is literally `0000000` |
| MQ55M | 2011-03-15 to 2011-04-04 | 349 | -34.05 / 150.67 | -54.5 / 159.0 | CADI `NAME Macquarie_Island` carrying Camden's coordinates |
| RL052 | 2024-06-07 to 2024-08-22 | 10,606 | 51.5 / 359.4 | 51.6 / 358.7 | Slough's coordinates on Chilton data |

ND51R's longitude, -169.93, does not fit the SAO F7.3 field and is written `*******`, so all 459
ND51R rows have a null longitude and the code is missing from `station.parquet`. ND61R's name
also appears once as `NiuePort_Moresby`.

## Unresolved

| code | period | rows | position | what the files say |
|---|---|---|---|---|
| HAJ43 | 1998-12-04 to 1999-11-22 | 15,398 | 37.9 / 284.5 (Wallops) | `DGS-256 242/HAJ43` with WMO ID KWLI (Wallops); no twins in WP937. Either the Hanscom instrument ran at Wallops, or its configuration was copied. The code's median position is Hanscom (42.45/288.70). |
| TMJ20 | 2016-03-17 to 2016-03-18 | 1,115 | -11.95 / 283.127 (Jicamarca) | a VIPIR declaring `/TMJ20_` |

Small groups at another station's coordinates, 1 to 74 rows each, none twinned except one:
AU930 (1 row at 42.45/288.8), BD840 (7, Wallops), EI764 (12, Pt Arguello), GM037 (61, Rome),
HAJ43 (27 at Ascension, one twinned with AS00Q; 13 at 37.1/127.0), HAJ45 (2, Wallops),
MA155 (9 at 55.8/38.3, 1 at 47.2/39.6), PE43K (1, Townsville), PRJ18 (1 at 18.0/255.0),
VT139 (1 at 41.2/284.6), WI937 (6, Boulder), WP937 (74 at 41.2/284.6, 1 Boulder, 1 PRJ18).

## Declared code differs, but not misfiled

Found by sampling one file per ten days across every station-year. These have no twins and carry
their directory station's coordinates.

- VT139 2001-02 files declare `WP999`, which is not a station.
- Single sampled files: EA036 declaring PA836 and VT139, EI764 declaring PA836, HAJ43 declaring
  AS00Q, HAJ45 declaring WI937, LM42B declaring THJ77, OK426 declaring VT139, PQ052 declaring
  FF051, GSJ53 and VT139, WP937 declaring WP999.

So a file's declared code alone is not a safe basis for relabelling.

## Coincidences, not duplicates

GM037 and RA041 (Gibilmanna and Rome, 452 km apart) share 412 soundings with identical foF2, MUF(D)
and hmF2 between 2018 and 2024. Each file names its own station, and the shared value sits a
median 0.05 MHz from its neighbours in both series, where the two stations normally differ by
0.1 MHz. Autoscala reports foF2 to 0.1 MHz with hmF2 often at a 200 km floor, so this is the rate
chance gives. MA155's 47 scattered matches with GM037, RA041 and MZ152 are the same.

## Reproducing

Positions per code, with distance from the main position:

```sql
CREATE VIEW mids AS SELECT * FROM read_parquet('/kass/forecast/mids-parquet/year=*/*.parquet');
WITH pos AS (
  SELECT code, lat, lon, ((lon % 360) + 360) % 360 AS lon360, count(*) AS n,
         min(time) AS t0, max(time) AS t1
  FROM mids WHERE lat IS NOT NULL AND lon IS NOT NULL GROUP BY ALL),
main AS (SELECT code, arg_max(lat, n) AS mlat, arg_max(lon360, n) AS mlon FROM pos GROUP BY code)
SELECT p.code, p.lat, p.lon, p.n, p.t0, p.t1,
       round(6371 * 2 * asin(sqrt(pow(sin(radians(p.lat - m.mlat) / 2), 2)
         + cos(radians(p.lat)) * cos(radians(m.mlat)) * pow(sin(radians(p.lon360 - m.mlon) / 2), 2)))) AS km
FROM pos p JOIN main m USING (code)
WHERE km >= 25 ORDER BY code, n DESC;
```

The same sounding under two codes:

```sql
WITH s AS (SELECT code, time, fof2, mufd, hmf2 FROM mids
           WHERE fof2 IS NOT NULL AND mufd IS NOT NULL AND hmf2 IS NOT NULL)
SELECT a.code, b.code, count(*) AS pairs, min(a.time), max(a.time)
FROM s a JOIN s b
  ON a.code < b.code AND a.fof2 = b.fof2 AND a.mufd = b.mufd AND a.hmf2 = b.hmf2
 AND b.time BETWEEN a.time - INTERVAL 2 MINUTE AND a.time + INTERVAL 2 MINUTE
GROUP BY ALL HAVING pairs >= 5 ORDER BY pairs DESC;
```

The second query is a large self-join; bucketing on `epoch(time) // 60` and joining neighbouring
buckets keeps it within a few GB.
