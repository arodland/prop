FROM python:latest
RUN apt-get update && apt-get install -y dumb-init python3-dev

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install git+https://github.com/hckr/geodesy.git

COPY ./app /app

ENTRYPOINT ["/usr/bin/dumb-init", "--verbose", "--"]

WORKDIR /app
CMD uwsgi --http 127.0.0.1:${RAYTRACE_PORT} --master --need-app --single-interpreter --max-requests 100 --http-timeout 300 --wsgi main:app --processes 16
