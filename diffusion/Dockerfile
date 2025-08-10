FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get -y install unzip build-essential cmake dumb-init libgfortran5 gfortran curl git

WORKDIR /

RUN curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=/usr/local bash

RUN pixi init pixi_env
COPY pixi.toml /pixi_env
WORKDIR /pixi_env
RUN pixi install

ENV PATH /pixi_env/.pixi/envs/default/bin:$PATH

COPY --from=iri2020 /build/ /build/

ENV HOME=/tmp

USER 1000
RUN huggingface-cli download --repo-type dataset arodland/IRI-iono-maps # 138182c

COPY app /app

# ENV OPENBLAS_NUM_THREADS=4
# ENV OMP_NUM_THREADS=4

ENTRYPOINT ["/usr/bin/dumb-init", "--"]
WORKDIR /app
CMD uwsgi --http :${DIFFUSION_PORT} --master --lazy --single-interpreter --http-timeout 600 --wsgi server:app --processes 4 --enable-metrics --carbon-use-metrics --carbon-name-resolve --carbon ${STATSD_HOST}:2003 --carbon-id prop-pred4d --die-on-term
