FROM continuumio/miniconda3

RUN apt-get update && apt-get -y install unzip build-essential

WORKDIR /
RUN conda create -y -q -n my_cartopy_env -c conda-forge python=3.7 statsmodels pandas cython xarray sympy networkx george

ENV PATH /opt/conda/envs/my_cartopy_env/bin:$PATH

RUN echo "conda activate my_cartopy_env" >> ~/.bashrc

COPY requirements.txt ./
RUN pip install -r requirements.txt

WORKDIR /

COPY app /app

CMD python3 /app/assimilate.py
