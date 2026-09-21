#!/bin/sh
# Mirror `pskreporter.rx` from wd10 (primary serving replica: no heavy queries, just stream) as one
# parquet per day, capped at ~100 Mbit/s. Full rows (the table has no id column).
#   nohup sh data/fetch_pskreporter.sh /kass/forecast/spots/pskreporter > /kass/forecast/spots/pskreporter/fetch.log 2>&1 &
# Idempotent: skips days already present. Only days strictly before today are fetched.
set -eu
out=$1; host=10.112.0.10; mkdir -p "$out"
d=${2:-2024-12-01}
stop=$(date -u +%F)
while [ "$d" != "$stop" ]; do
    next=$(date -u -d "$d + 1 day" +%F)
    f="$out/$d.parquet"
    if [ ! -s "$f" ]; then
        echo "$(date -u +%FT%TZ) $d"
        ssh -o BatchMode=yes "$host" "clickhouse-client -d pskreporter --output_format_parquet_compression_method=zstd --query \"SELECT * FROM rx WHERE time >= '$d' AND time < '$next' FORMAT Parquet\"" \
            | pv -q -L 12M > "$f.tmp" && mv "$f.tmp" "$f" || { rm -f "$f.tmp"; echo "FAILED $d"; sleep 60; continue; }
    fi
    d=$next
done
echo done
