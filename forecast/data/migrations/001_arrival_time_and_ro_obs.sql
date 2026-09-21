-- PLAN.md Phase 0: make future data better than past data.
-- Run once against the prop DB (psql inside the pod).

-- Arrival time, so point-in-time replay can use real latency instead of an assumed 15 min.
ALTER TABLE measurement ADD COLUMN IF NOT EXISTS inserted_at timestamptz NOT NULL DEFAULT now();

-- Raw RO observations, independent of any model run (cosmic_eval only stores comparison rows).
CREATE TABLE IF NOT EXISTS ro_obs (
    time        timestamp        NOT NULL,
    latitude    double precision NOT NULL,
    longitude   double precision NOT NULL,
    fof2        double precision,
    hmf2        double precision,
    source      text             NOT NULL,
    inserted_at timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (time, latitude, longitude, source)
);
CREATE INDEX IF NOT EXISTS ro_obs_time_idx ON ro_obs (time);
