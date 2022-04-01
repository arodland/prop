FROM iri2016 AS essn
RUN apt-get update && apt-get install -y python3 python3-dateutil python3-pip python3-scipy python3-numpy git dumb-init
WORKDIR /src
COPY requirements.txt ./
RUN pip3 install -r requirements.txt
COPY heval.py ./

ENTRYPOINT ["/usr/bin/dumb-init", "--verbose", "--"]
CMD uwsgi --http :${HEVAL_PORT} --master --need-app --single-interpreter --enable-threads --max-requests 100 --http-timeout 300 --wsgi heval:app --processes 2
