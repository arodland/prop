FROM continuumio/miniconda3  
  
RUN apt-get update && apt-get -y install unzip build-essential nodejs npm dumb-init libhdf5-dev
RUN npm install -g svgo

RUN conda create -y -q -n my_cartopy_env -c conda-forge python=3.10 matplotlib=3.6.0 cartopy=0.21.0 pyproj=3.4.1 pandas h5py=3.7.0 scipy pykdtree uwsgi
  
ENV PATH /opt/conda/envs/my_cartopy_env/bin:$PATH  
  
RUN echo "conda activate my_cartopy_env" >> ~/.bashrc  
  
COPY requirements.txt ./
RUN pip install -r requirements.txt

RUN cartopy_feature_download.py physical -o /app/cartopy_data --no-warn

WORKDIR /  
COPY app /app  

ENTRYPOINT ["/usr/bin/dumb-init", "--verbose", "--"]
WORKDIR /app
ENV HOME=/
ENV OPENBLAS_NUM_THREADS=4
ENV OMP_NUM_THREADS=4
CMD uwsgi --http :${RENDERER_PORT} --master --need-app --single-interpreter --max-requests 100 --http-timeout 300 --wsgi renderer:app --processes 32 --die-on-term
