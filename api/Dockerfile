FROM python:latest
RUN apt-get update && apt-get install -y dumb-init
ENV STATIC_INDEX 1

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

COPY ./app /app

ENTRYPOINT ["/usr/bin/dumb-init", "--verbose", "--"]
WORKDIR /app
CMD uwsgi --http :${API_PORT} --master --need-app --single-interpreter --max-requests 100 --http-timeout 300 --wsgi main:app --processes 32
