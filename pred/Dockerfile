FROM continuumio/miniconda3

RUN apt-get update && apt-get -y install unzip build-essential

WORKDIR /

RUN conda create -y -q -n my_cartopy_env -c conda-forge python=3.7 statsmodels pandas cython xarray sympy networkx scikit-learn

ENV PATH /opt/conda/envs/my_cartopy_env/bin:$PATH

RUN echo "conda activate my_cartopy_env" >> ~/.bashrc

RUN pip install -U pip
RUN pip install Flask==1.0.2 Werkzeug==0.14.1 pyyaml pybind11

WORKDIR /src
RUN git clone https://github.com/arodland/george
WORKDIR /src/george
RUN python setup.py build
RUN python setup.py install

WORKDIR /
COPY app /app

CMD python /app/server.py
