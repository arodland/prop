FROM debian:buster-slim AS build
RUN apt-get update && apt-get install -y gfortran build-essential meson
COPY src /src
WORKDIR /build
RUN meson /src
RUN ninja
COPY data/ /build/iri2016/data/

FROM debian:buster-slim AS iri2016
RUN apt-get update && apt-get install -y gfortran
COPY --from=build /build /build
