FROM python:3.9
RUN apt-get update && apt-get install -y dumb-init python3-dev libhdf5-dev

RUN pip3 install numpy==1.21.5

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt
RUN uwsgi --build-plugin https://github.com/KLab/uwsgi-cheaper-spare2

COPY ./app /app
RUN mv /cheaper_spare2_plugin.so /app

ENTRYPOINT ["/usr/bin/dumb-init", "--verbose", "--"]

WORKDIR /app
CMD uwsgi --http 127.0.0.1:${RAYTRACE_PORT} --master --need-app --single-interpreter --max-requests 100 --http-timeout 300 --wsgi main:app --processes 64 --need-plugin cheaper_spare2 --cheaper-algo spare2 --cheaper 8 --cheaper-step 8 --cheaper-initial 8
