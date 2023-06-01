FROM iri2020 AS essn
RUN apt-get update && apt-get install -y python3 python3-pip git dumb-init
WORKDIR /src
COPY requirements.txt ./
RUN pip3 install -r requirements.txt
COPY essn.py ./

ENTRYPOINT ["/usr/bin/dumb-init", "--verbose", "--"]
CMD uwsgi --http :${ESSN_PORT} --master --need-app --single-interpreter --enable-threads --max-requests 100 --http-timeout 300 --wsgi essn:app --processes 2 --enable-metrics --carbon-use-metrics --carbon-name-resolve --carbon ${STATSD_HOST}:2003 --carbon-id prop-essn --die-on-term
