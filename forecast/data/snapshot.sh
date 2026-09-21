#!/bin/sh
# Dump the tables the forecast project needs to gzipped CSV, from inside the prop pod.
#
# Usage (on prop.kc2g.com):
#   podman run --rm --pod prop --env-file /etc/kc2gprop/db.env \
#     --mount type=bind,src=/path/to/nfs/snapshot,dst=/out \
#     -v "$PWD/snapshot.sh:/snapshot.sh:ro" postgres:16 sh /snapshot.sh
#
# Output: /out/<date>/<table>.csv.gz — full tables, unfiltered. Trimming happens in to_parquet.py.
set -eu

out=/out/$(date -u +%Y-%m-%d)
mkdir -p "$out"
export PGPASSWORD="$DB_PASSWORD"

dump() { # dump <name> <sql>
    echo "$1"
    psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -qAt \
        -c "COPY ($2) TO STDOUT WITH (FORMAT csv, HEADER)" | gzip > "$out/$1.csv.gz.tmp"
    mv "$out/$1.csv.gz.tmp" "$out/$1.csv.gz"
}

dump station     "SELECT * FROM station"
dump measurement "SELECT * FROM measurement"
dump cosmic_eval "SELECT * FROM cosmic_eval"
# Live-experiment scoring rows: the fixed reference for the production GP (PLAN.md Phase 2, baseline 5).
dump holdout     "SELECT * FROM holdout"
dump holdout_eval "SELECT * FROM holdout_eval"
dump pred_eval   "SELECT * FROM pred_eval"
dump runs        "SELECT * FROM runs"
# Production eSSN fits (per run, series 24h/6h): honest as-of-T driver for the IRI+eSSN baseline.
dump essn        "SELECT * FROM essn"

echo done: "$out"
