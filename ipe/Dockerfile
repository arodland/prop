FROM iri2020 AS ipe
RUN apt-get update && apt-get install -y python3 python3-pip git dumb-init libhdf5-dev libnetcdf-dev
WORKDIR /src
COPY requirements.txt ./
RUN pip3 install -r requirements.txt
COPY ipe.py ./

ENTRYPOINT ["/usr/bin/dumb-init", "--verbose", "--"]
CMD uwsgi --http :${IPE_PORT} --master --need-app --single-interpreter --enable-threads --max-requests 100 --http-timeout 300 --wsgi ipe:app --processes 8 --enable-metrics --carbon-use-metrics --carbon-name-resolve --carbon ${STATSD_HOST}:2003 --carbon-id prop-ipe --die-on-term
