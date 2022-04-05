FROM python:3.7
RUN apt-get update && apt-get install -y dumb-init

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

COPY ./app /app

ENTRYPOINT ["/usr/bin/dumb-init", "--verbose", "--"]
CMD python3 /app/main.py
