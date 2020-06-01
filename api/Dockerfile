FROM python:latest
RUN apt-get update && apt-get install -y dumb-init
ENV STATIC_INDEX 1

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

COPY ./app /app

ENTRYPOINT ["/usr/bin/dumb-init", "--verbose", "--"]
CMD python3 /app/main.py
