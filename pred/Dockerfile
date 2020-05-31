FROM continuumio/miniconda3

RUN apt-get update && apt-get -y install unzip build-essential dumb-init

WORKDIR /

RUN conda create -y -q -n my_cartopy_env -c conda-forge python=3.7 statsmodels pandas cython xarray sympy networkx scikit-learn

ENV PATH /opt/conda/envs/my_cartopy_env/bin:$PATH
ENV PYTHON_EGG_CACHE=/tmp/egg-cache

RUN echo "conda activate my_cartopy_env" >> ~/.bashrc

RUN pip install -U pip
COPY requirements.txt ./
RUN pip install -r requirements.txt

WORKDIR /src
RUN git clone https://github.com/arodland/george
WORKDIR /src/george
RUN python setup.py build
RUN python setup.py install

WORKDIR /
COPY app /app

ENTRYPOINT ["/usr/bin/dumb-init", "--verbose", "--"]
CMD python /app/server.py
