FROM tiangolo/meinheld-gunicorn-flask:python3.7

ENV STATIC_INDEX 1

COPY requirements.txt /tmp/

RUN pip install -U pip

RUN pip install -r /tmp/requirements.txt

COPY ./app /app
