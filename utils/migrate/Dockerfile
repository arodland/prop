FROM perl:5 AS perl-cpm
RUN cpanm --notest App::cpm

FROM perl-cpm AS final
# RUN apt-get update && apt-get install dumb-init

COPY cpanfile .
RUN cpm install -g

COPY ./app /app
COPY ./lib /perl5lib

# ENTRYPOINT ["/usr/bin/dumb-init", "--verbose", "--"]
CMD /app/migrate.pl
