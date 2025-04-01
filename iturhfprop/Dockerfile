FROM perl:5.38.2 AS perl-cpm
RUN wget -O /usr/local/bin/cpm https://raw.githubusercontent.com/skaji/cpm/0.997015/cpm && chmod +x /usr/local/bin/cpm

FROM perl-cpm AS final
RUN apt-get update && apt-get install dumb-init

COPY ITU-R-HF/ITURHFProp/Linux/ITURHFProp /usr/local/bin
COPY ITU-R-HF/ITURHFProp/Linux/libp372.so /usr/local/lib
COPY ITU-R-HF/ITURHFProp/Linux/libp533.so /usr/local/lib
COPY ITU-R-HF/ITURHFProp/Data /opt/iturhfprop/data
RUN ldconfig

COPY cpanfile .
RUN cpm install -g
COPY ./app /app

ENTRYPOINT ["/usr/bin/dumb-init", "--verbose", "--"]
ENV MOJO_REVERSE_PROXY=1
CMD perl -I/app/lib /app/main.pl prefork -m production -w 16 -s 16 -c 2 -l "http://*:$ITURHFPROP_PORT"
# CMD perl -I/app/lib /app/main.pl daemon -l "http://*:$ITURHFPROP_PORT"
