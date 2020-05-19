FROM perl:5 AS perl-cpm
RUN cpanm --notest App::cpm

FROM perl-cpm AS final
RUN apt-get update && apt-get install inoticoming dumb-init

COPY cpanfile .
RUN cpm install -g

COPY ./app /app
COPY ./bin /noaa-loader
COPY ./lib /perl5lib

ENTRYPOINT ["/usr/bin/dumb-init", "--"]
CMD ["/noaa-loader/listen"]
