# prop-forecast: the anomaly model as a production map producer. CUDA runtime for the A4500; CPU fallback works.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv curl ca-certificates libhdf5-dev libnetcdf-dev dumb-init && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra train --no-dev
COPY baselines ./baselines
COPY data ./data
COPY train ./train
COPY service ./service
COPY analysis/wspr_aggregate.py ./analysis/wspr_aggregate.py
ENV FORECAST_MODELS=/checkpoints IRI_CACHE=/checkpoints/iri_cache GLOTEC_DIR=/checkpoints/glotec OMP_NUM_THREADS=8
EXPOSE 5515
ENTRYPOINT ["/usr/bin/dumb-init", "--"]
CMD ["/app/.venv/bin/python", "/app/service/app.py"]
