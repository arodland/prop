FROM perl:5.38.2 AS perl-base

FROM perl-base AS perl-cpm
RUN wget -O /usr/local/bin/cpm https://raw.githubusercontent.com/skaji/cpm/0.997015/cpm && chmod +x /usr/local/bin/cpm

FROM perl-cpm as build-iturhfprop

RUN apt-get update && apt-get install build-essential

COPY ITU-R-HF /src

WORKDIR /src/P372/Linux
RUN make OPTIMIZE='-O3 -flto -march=native'

WORKDIR /src/P533/Linux
RUN make OPTIMIZE='-O3 -flto -march=native'

WORKDIR /src/ITURHFProp/Linux
RUN make OPTIMIZE='-O3 -flto -march=native'

FROM perl-cpm AS perl-deps
COPY cpanfile .
RUN cpm install -g

FROM perl-base AS final
RUN apt-get update && apt-get install dumb-init

COPY --from=build-iturhfprop /src/ITURHFProp/Linux/ITURHFProp /usr/local/bin
COPY --from=build-iturhfprop /src/P372/Linux/libp372.so /usr/local/lib
COPY --from=build-iturhfprop /src/P533/Linux/libp533.so /usr/local/lib
COPY ITU-R-HF/ITURHFProp/Data /opt/iturhfprop/data
RUN ldconfig

COPY --from=perl-deps /usr/local/lib/perl5 /usr/local/lib/perl5
COPY ./app /app

ENTRYPOINT ["/usr/bin/dumb-init", "--verbose", "--"]
ENV MOJO_REVERSE_PROXY=1
CMD perl -I/app/lib /app/main.pl prefork -m production -w 2 -s 16 -l "http://*:$ITURHFPROP_PORT"
# CMD perl -I/app/lib /app/main.pl daemon -l "http://*:$ITURHFPROP_PORT"
