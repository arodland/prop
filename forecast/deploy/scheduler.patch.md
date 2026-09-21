# Scheduler changes for the forecast_v2 experiments (apply to the server's scheduler/app/main.pl)

1. `use Task::Forecast;` next to the other Task plugins, and copy `Task-Forecast.pm` to `lib/Task/Forecast.pm`.
2. Add `FORECAST_PORT=5515` to `/etc/kc2gprop/ports.env` (and `etc/ports.env`).
3. In `one_run`, next to the `if ($jobs->{forecast_diffusion}) { ... }` block after the target-time loop, add:

```perl
  if ($jobs->{forecast_v2}) {
    my $forecast = app->minion->enqueue('forecast_v2',
      [
        run_id  => $run_id,
        model   => $jobs->{forecast_v2}{model},
        glotec  => ($jobs->{forecast_v2}{glotec} ? 1 : 0),
        spots   => ($jobs->{forecast_v2}{spots} ? 1 : 0),
        holdout => ($jobs->{holdout_all_timestep} ? 1 : 0),
      ],
      {
        parents  => [],          # needs only the run row, which is inserted above
        attempts => 2,
        expire   => 18 * 60,
      },
    );
    push @iongrid_deps, $forecast;
    push @band_quality_deps, $forecast;
    for my $render (@target_times) {
      $map_deps{$render->{target_time}} = $forecast;
    }
  }
```

   and change the `elsif (!$jobs->{forecast_diffusion})` guard on the per-target `assimilate` enqueue to
   `elsif (!$jobs->{forecast_diffusion} && !$jobs->{forecast_v2})`. The per-target `irimap` jobs may also be
   skipped when `$jobs->{forecast_v2}` is set (they are only needed by assimilate/diffusion).

4. Experiments in `queue_job`:

```perl
    sub {
      one_run($run_time, $state, '2026-09-v4', {
          forecast_v2 => { model => 'v4', glotec => 0 },
          make_maps => 1, renderhtml => 1, no_holdout => 1,
      });
    },
    sub {
      one_run($run_time, $state, '2026-09-v4-glotec', {
          forecast_v2 => { model => 'v4', glotec => 1 },
          make_maps => 1, renderhtml => 1, no_holdout => 1,
      });
    },
    # v9b_spatial_3 (query state + spatial dropout, spots, no GloTEC) with and without live spot tokens; replaces the v7p6 pair
    sub {
      one_run($run_time, $state, '2026-09-sp3', {
          forecast_v2 => { model => 'v9b_spatial_3', glotec => 0, spots => 0 },
          make_maps => 1, renderhtml => 1, no_holdout => 1,
      });
    },
    sub {
      one_run($run_time, $state, '2026-09-sp3-spots', {
          forecast_v2 => { model => 'v9b_spatial_3', glotec => 0, spots => 1 },
          make_maps => 1, renderhtml => 1, no_holdout => 1,
      });
    },
    # v7p6 side by side with and without live spot tokens (spots => 1 needs prop-spots.timer running)
    sub {
      one_run($run_time, $state, '2026-09-v7p6', {
          forecast_v2 => { model => 'v7p6', glotec => 1, spots => 0 },
          make_maps => 1, renderhtml => 1, no_holdout => 1,
      });
    },
    sub {
      one_run($run_time, $state, '2026-09-v7p6-spots', {
          forecast_v2 => { model => 'v7p6', glotec => 1, spots => 1 },
          make_maps => 1, renderhtml => 1, no_holdout => 1,
      });
    },
```

   The production run stays as the control. Checkpoints live at `/home/prop/checkpoints/<model>/best.pt`.

# Daily indices (deploy/prop-indices.{service,timer})

`prop-indices.service` runs `service/indices.py` from the `prop-forecast` image once a day (03:30 UTC) and upserts
GFZ Kp/ap/F10.7 into `indices_daily` and SILSO monthly SSN into `ssn_monthly`, creating both tables if missing.
The forecast service uses `indices_daily` for its trailing-81-day F10.7 driver and global token; without it, it falls
back to the latest eSSN `sfi` and a quiet ap. Run it once by hand after installing so the first forecast has indices.
The forecast service also records `glotec_latency_min` in each map file's attributes (age of the newest GloTEC step at
issue time); training assumed 30 min.

# Keeping experiment runs (Task/Cleanup.pm)

`archive_run` moves every run (experiments included) from the DB to `/archive/<id>` after 3 days, but after
14 days `upload_run` only ships production runs offsite (`/offsite/prop-archive/...`, the tree scored by
`eval/gp_maps.py`); experiment runs are deleted. To keep the forecast_v2 experiments scorable, change the
gate in the 14-day loop from

    if (!defined $run->{experiment}) { upload } else { delete }

to

    if (!defined $run->{experiment} || $run->{experiment} =~ /^2026-09-v4/) { upload } else { delete }

`eval/gp_maps.py` reads either layout (`PROP_ARCHIVE=/archive` on the server for the flat pre-upload tree,
default the offsite id-sharded tree) and takes `--experiment <name>`.

## Live spot tokens (2026-09-08)

1. `mkdir -p /home/prop/checkpoints/spots` and copy `/kass/forecast/eval/spot_baseline_cp.parquet` into it (once;
   rebuild it only when the training baseline changes).
2. Rebuild `prop-forecast` (the image now carries `analysis/wspr_aggregate.py` and `service/spots.py`).
3. Install `deploy/prop-spots.{service,timer}`, `systemctl enable --now prop-spots.timer`, then
   `systemctl start prop-spots.service` once by hand: the first run backfills 26 h (~30 s), later runs pull
   15 min (~3 s WSPR, ~15 s FT8 incl. re-aggregation). Layout under `/home/prop/checkpoints/spots/`:
   `raw/{wspr,psk}/` (3-day retention), `agg/*_hourly_live.parquet`, `watermark.json`.
4. Restart `prop-forecast`. A spots-capable model (`f_spot` in the checkpoint) then builds spot tokens
   whenever `agg/` has files; run logs show `spots: N tokens, newest complete hour ended M min before T`
   and the HDF5 gets `n_spot_tokens` / `spot_latency_min` attrs. `FORECAST_SPOT_CAP` (default 6000) must
   match the cap the checkpoint's samples were built with (v7p6: 6000; v7p: 3000).
