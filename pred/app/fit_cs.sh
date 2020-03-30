#!/bin/bash
count="$1"
for x in `seq 1 "$count"` ; do
  echo "$x" 1>&2
  python /app/fit_cs.py
  sleep 0.1
done

