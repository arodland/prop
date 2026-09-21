#!/bin/sh
# Mirror the wsprdaemon `wspr.rx` table from wd20 as one parquet per month, capped at ~100 Mbit/s.
#   nohup sh data/fetch_wspr.sh /kass/forecast/spots/wspr > /kass/forecast/spots/wspr/fetch.log 2>&1 &
# Full copy including `id`, so the table can be re-seeded exactly. Idempotent: skips months already present. Only months strictly before the current one are fetched.
set -eu
out=$1; host=10.112.0.20; mkdir -p "$out"
m=${2:-2008-03-01}
stop=$(date -u +%Y-%m-01)
while [ "$m" != "$stop" ]; do
    next=$(date -u -d "$m + 1 month" +%Y-%m-01)
    f="$out/$(date -d "$m" +%Y-%m).parquet"
    if [ ! -s "$f" ]; then
        echo "$(date -u +%FT%TZ) $m"
        ssh -o BatchMode=yes "$host" "clickhouse-client -d wspr --output_format_parquet_compression_method=zstd --query \"SELECT * FROM rx WHERE time >= '$m' AND time < '$next' FORMAT Parquet\"" \
            | pv -q -L 12M > "$f.tmp" && mv "$f.tmp" "$f" || { rm -f "$f.tmp"; echo "FAILED $m"; sleep 60; continue; }
    fi
    m=$next
done
echo done
