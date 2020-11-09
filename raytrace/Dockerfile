FROM python:latest
RUN apt-get update && apt-get install -y dumb-init python3-dev

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
RUN pip install git+https://github.com/hckr/geodesy.git

COPY ./app /app

# ENTRYPOINT ["/usr/bin/dumb-init", "--verbose", "--"]
CMD python3 /app/main.py
