FROM perl:5 AS perl-cpm
RUN cpanm --notest App::cpm

FROM perl-cpm AS final
RUN apt-get update && apt-get install dumb-init

COPY cpanfile .
RUN cpm install -g

COPY ./app /app

ENTRYPOINT ["/usr/bin/dumb-init", "--verbose", "--"]
CMD perl -I/app/lib /app/main.pl daemon -l "http://*:$SCHEDULER_PORT"
