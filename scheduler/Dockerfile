FROM perl:5.38.2 AS perl-cpm
RUN wget -O /usr/local/bin/cpm https://raw.githubusercontent.com/skaji/cpm/0.997015/cpm && chmod +x /usr/local/bin/cpm

FROM perl-cpm AS final
RUN apt-get update && apt-get install dumb-init

COPY cpanfile .
RUN cpm install -g

COPY ./app /app

ENTRYPOINT ["/usr/bin/dumb-init", "--verbose", "--"]
CMD perl -I/app/lib /app/main.pl daemon -l "http://*:$SCHEDULER_PORT"
