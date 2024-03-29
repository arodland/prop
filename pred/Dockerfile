FROM continuumio/miniconda3

RUN apt-get update && apt-get -y install unzip build-essential dumb-init libgfortran5

WORKDIR /

RUN conda create -y -q --insecure -n my_cartopy_env -c conda-forge python=3.10 statsmodels pandas cython xarray sympy networkx scikit-learn scikit-optimize uwsgi george

ENV PATH /opt/conda/envs/my_cartopy_env/bin:$PATH
ENV PYTHON_EGG_CACHE=/tmp/egg-cache

COPY --from=iri2020 /build/ /build/

RUN echo "conda activate my_cartopy_env" >> ~/.bashrc

RUN pip install -U pip
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app /app

ENV OPENBLAS_NUM_THREADS=4
ENV OMP_NUM_THREADS=4

ENTRYPOINT ["/usr/bin/dumb-init", "--verbose", "--"]
WORKDIR /app
CMD uwsgi --http :${PRED_PORT} --master --need-app --single-interpreter --enable-threads --max-requests 100 --http-timeout 300 --wsgi server:app --processes 6 --enable-metrics --carbon-use-metrics --carbon-name-resolve --carbon ${STATSD_HOST}:2003 --carbon-id prop-pred --die-on-term
