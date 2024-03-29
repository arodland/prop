FROM continuumio/miniconda3

RUN apt-get update && apt-get -y install unzip build-essential dumb-init

WORKDIR /
RUN conda create -y -q -n my_cartopy_env -c conda-forge python=3.10 statsmodels pandas cython xarray sympy networkx uwsgi george

ENV PATH /opt/conda/envs/my_cartopy_env/bin:$PATH

RUN echo "conda activate my_cartopy_env" >> ~/.bashrc

COPY requirements.txt ./
RUN pip install -r requirements.txt

WORKDIR /

COPY app /app

ENTRYPOINT ["/usr/bin/dumb-init", "--verbose", "--"]
WORKDIR /app
ENV OPENBLAS_NUM_THREADS=4
ENV OMP_NUM_THREADS=4

CMD uwsgi --http :${ASSIMILATE_PORT} --master --need-app --single-interpreter --max-requests 100 --http-timeout 300 --wsgi assimilate:app --processes 8 --enable-metrics --carbon-use-metrics --carbon-name-resolve --carbon ${STATSD_HOST}:2003 --carbon-id prop-assimilate --die-on-term
