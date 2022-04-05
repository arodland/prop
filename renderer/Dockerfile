FROM continuumio/miniconda3  
  
RUN apt-get update && apt-get -y install unzip build-essential nodejs npm dumb-init libhdf5-dev
RUN npm install -g svgo

WORKDIR /  
RUN wget -O /tmp/ne_110m_land.zip http://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip  
WORKDIR /.local/share/cartopy/shapefiles/natural_earth/physical  
RUN unzip /tmp/ne_110m_land.zip && rm /tmp/ne_110m_land.zip  
  
RUN conda create -y -q -n my_cartopy_env -c conda-forge python=3.9 matplotlib=3.3.3 cartopy=0.18.0 proj=7.1.1 pandas h5py=2.10.0 scipy pykdtree uwsgi
  
ENV PATH /opt/conda/envs/my_cartopy_env/bin:$PATH  
  
RUN echo "conda activate my_cartopy_env" >> ~/.bashrc  
  
COPY requirements.txt ./
RUN pip install -r requirements.txt

WORKDIR /  
COPY app /app  

ENTRYPOINT ["/usr/bin/dumb-init", "--verbose", "--"]
WORKDIR /app
ENV HOME=/
CMD uwsgi --http :${RENDERER_PORT} --master --need-app --single-interpreter --max-requests 100 --http-timeout 300 --wsgi renderer:app --processes 32
