FROM iri2020 AS cosmic
RUN apt-get update && apt-get install -y python3 python3-pip git dumb-init libnetcdf-dev
WORKDIR /src
COPY requirements.txt ./
RUN pip3 install -r requirements.txt
COPY cosmic.py ./
COPY backfill_hmf2.py ./
COPY backfill_irtam.py ./

ENTRYPOINT ["/usr/bin/dumb-init", "--verbose", "--"]
CMD python3 -u /src/cosmic.py

