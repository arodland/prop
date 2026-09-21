#!/bin/sh
# Mirror NOAA GloTEC daily netCDF files. Idempotent; rerun to pick up new days.
#   sh data/fetch_glotec.sh /kass/forecast/glotec [start YYYY-MM-DD]
set -eu
out=$1
start=${2:-2025-05-10}  # NOAA archive begins mid-May 2025; ~15 MB/day
mkdir -p "$out"
d=$start
while [ "$d" != "$(date -u -d 'tomorrow' +%F)" ]; do
    f="GloTEC_TEC_$(date -d "$d" +%Y_%m_%d).nc"
    [ -s "$out/$f" ] || curl -sf -o "$out/$f.tmp" "https://services.swpc.noaa.gov/products/glotec/netcdf_2d_urt/$f" && mv -f "$out/$f.tmp" "$out/$f" 2>/dev/null || rm -f "$out/$f.tmp"
    d=$(date -d "$d + 1 day" +%F)
done
ls "$out" | wc -l
