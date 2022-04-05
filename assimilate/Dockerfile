FROM continuumio/miniconda3

RUN apt-get update && apt-get -y install unzip build-essential dumb-init

WORKDIR /
RUN conda create -y -q -n my_cartopy_env -c conda-forge python=3.7 statsmodels pandas cython xarray sympy networkx uwsgi george

ENV PATH /opt/conda/envs/my_cartopy_env/bin:$PATH

RUN echo "conda activate my_cartopy_env" >> ~/.bashrc

COPY requirements.txt ./
RUN pip install -r requirements.txt

WORKDIR /

COPY app /app

ENTRYPOINT ["/usr/bin/dumb-init", "--verbose", "--"]
WORKDIR /app
CMD uwsgi --http :${ASSIMILATE_PORT} --master --need-app --single-interpreter --max-requests 100 --http-timeout 300 --wsgi assimilate:app --processes 8
