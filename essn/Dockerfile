FROM iri2016 AS essn
RUN apt-get update && apt-get install -y python3 python3-dateutil python3-xarray python3-sklearn python3-pip python3-sympy python3-pybind11 cython3 git dumb-init
WORKDIR /src
COPY requirements.txt ./
RUN pip3 install -r requirements.txt
COPY essn.py ./

ENTRYPOINT ["/usr/bin/dumb-init", "--verbose", "--"]
CMD uwsgi --http :${ESSN_PORT} --master --need-app --single-interpreter --enable-threads --max-requests 100 --http-timeout 300 --wsgi essn:app --processes 2
