FROM iri2020 AS irimap
RUN apt-get update && apt-get install -y python3 python3-dateutil python3-xarray python3-sklearn python3-pip python3-sympy python3-pybind11 cython3 git dumb-init cmake
WORKDIR /src
COPY requirements.txt ./
RUN pip3 install -r requirements.txt
# wmm2020 does a build-on-first-use thing...
RUN python3 -c 'import wmm2020'
COPY irimap.py ./

ENTRYPOINT ["/usr/bin/dumb-init", "--verbose", "--"]
CMD uwsgi --http :${IRIMAP_PORT} --master --need-app --single-interpreter --max-requests 100 --http-timeout 300 --wsgi irimap:app --processes 16 --enable-metrics --carbon-use-metrics --carbon-name-resolve --carbon ${STATSD_HOST}:2003 --carbon-id irimap --die-on-term
