FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get -y install unzip build-essential cmake dumb-init libgfortran5 gfortran curl git python3 python3-dev

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN /root/.local/bin/uv tool install 'git+https://github.com/bluss/sysconfigpatcher'
RUN /root/.local/bin/uv venv --python 3.13 /app/.venv
RUN /root/.local/bin/sysconfigpatcher /root/.local/share/uv/python/cpython-3.13*
RUN chmod go+rx /root && chmod -R go+rx /root/.local

ENV PATH="/app/.venv/bin:$PATH"
COPY pyproject.toml /app/pyproject.toml

WORKDIR /app
RUN /root/.local/bin/uv sync

COPY --from=iri2020 /build/ /build/

ENV HOME=/tmp

USER 1000
RUN hf download --repo-type dataset arodland/IRI-iono-maps

COPY app /app

# ENV OPENBLAS_NUM_THREADS=4
# ENV OMP_NUM_THREADS=4

ENTRYPOINT ["/usr/bin/dumb-init", "--"]
WORKDIR /app
CMD uwsgi --http :${DIFFUSION_PORT} --master --lazy --single-interpreter --http-timeout 600 --wsgi server:app --processes 4 --enable-metrics --carbon-use-metrics --carbon-name-resolve --carbon ${STATSD_HOST}:2003 --carbon-id prop-pred4d --die-on-term
