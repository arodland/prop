#!/usr/bin/perl
use Mojolicious::Lite;
use Mojo::IOLoop;
use Mojo::Pg;
use Mojo::UserAgent;
use List::Util 'shuffle';

app->secrets([$ENV{MOJO_SECRET}]);

helper pg => sub {
  state $pg = Mojo::Pg->new("postgresql://$ENV{DB_USER}:$ENV{DB_PASSWORD}\@$ENV{DB_HOST}/$ENV{DB_NAME}");
};

plugin 'Minion' => {
  Pg => app->pg,
};

app->minion->missing_after(3 * 60);
app->minion->repair;

app->minion->on(dequeue => sub {
  my ($job) = @_;
  warn "[$$] ", $job->id, " dequeued\n";
  $job->on(finished => sub {
    my ($job) = @_;
    warn "[$$] ", $job->id, " finished\n";
  });
  $job->on(failed => sub {
    my ($job) = @_;
    warn "[$$] ", $job->id, " failed\n";
  });
});

plugin 'Minion::Admin';
plugin 'Minion::Notifier';
plugin 'Minion::Statsd';

plugin 'Task::eSSN';
plugin 'Task::Pred';
plugin 'Task::IRIMap';
plugin 'Task::IPE';
plugin 'Task::Assimilate';
plugin 'Task::BandQuality';
plugin 'Task::Render';
plugin 'Task::Cleanup';
plugin 'Task::HoldoutEvaluate';

sub prev_next {
  my $INTERVAL = 900; # 15 minutes
  my $LEAD = 300; # Run 5 minutes early, e.g. :10, :25, :40, :55

  my $now = time;
  my $prev = $now - (($now + $LEAD) % $INTERVAL);
  my $next = $prev + $INTERVAL;

  return ($prev, $next, $now);
}

sub next_run {
  my ($prev, $next, $now) = prev_next();
  my $wait = $next - $now;
  return ($next, $wait);
}

sub target_times {
  my ($run_time) = @_;

  return (
    {
      name => 'now',
      target_time => $run_time + 300,
      dots => 'curr',
    },
    map(+{
      name => "${_}h",
      target_time => $run_time + 300 + $_*3600,
      dots => 'pred',
    }, 1 .. 24),
  )
}

sub pred_times {
  my ($run_time) = @_;

  # Every 15 minutes from -1hr to +6hr; every hour from +7hr to +24hr, inclusive.
  return map({ $run_time + 300 + 900*$_ } -4 .. 24), map({ $run_time + 300 + 3600*$_ } 7 .. 24);
}


sub make_maps {
  my (%args) = @_;

  my @jobs;

  my %file_formats_by_format = (
    normal => [
      'svg',
      'station_json',
      'geojson',
    ],
    bare => [
      'jpg',
    ],
    overlay => [
      'svg',
    ],
  );

  for my $metric (qw(mufd fof2)) {
    for my $format (qw(normal bare)) {
      push @jobs, app->minion->enqueue('rendersvg',
        [
          run_id => $args{run_id},
          target => $args{target},
          metric => $metric,
          name   => $args{name},
          format => $format,
          dots   => $args{dots},
          file_format => $file_formats_by_format{$format},
        ],
        {
          parents => $args{parents},
          attempts => 2,
          expire => 18 * 60,
        }
      );
    }
  }

  for my $metric (qw(muf_sp)) {
    for my $loc (keys %Task::Render::LOCATIONS) {
      push @jobs, app->minion->enqueue('rendermuf',
        [
          $loc,
          run_id => $args{run_id},
          target => $args{target},
          metric => $metric,
          name => "$loc-$args{name}",
          res => 1,
        ],
        {
          parents => $args{parents},
          attempts => 2,
          expire => 18 * 60,
        },
      );
    }
  }

  return @jobs;
}

sub one_run {
  my ($run_time, $state, $experiment, $jobs) = @_;
  my @target_times = target_times($run_time);
  my $first_target_time = $target_times[0]{target_time};

  my $holdout_meas = $state->{holdout_meas};
  my $holdouts = @$holdout_meas && !$jobs->{no_holdout} && eval {
    Mojo::UserAgent->new->inactivity_timeout(30)->post("http://localhost:$ENV{API_PORT}/holdout", form => { measurements => $holdout_meas })->result->json
  } || [];

  my @holdout_ids = map $_->{holdout}, @$holdouts;
  my @holdout_times = map $_->{ts}, @$holdouts;

  my @stations = app->pg->db->query("select distinct(station_id) from measurement where time > now() - interval '14 day' and time <= now()")->hashes->map(sub { $_->{station_id} })->each;

  my $essn_24h = app->minion->enqueue('essn',
    [
      series => '24h',
      holdouts => [ @holdout_ids ],
      v2 => $jobs->{essn_v2},
    ],
    {
      attempts => 2,
      expire => 18 * 60,
    },
  );

  my $run_id = $essn_24h;
  app->pg->db->query('insert into runs (id, started, target_time, experiment, state) values (?, to_timestamp(?), to_timestamp(?), ?, ?)',
    $run_id, time(), $first_target_time, $experiment, 'created'
  );

  my $essn_6h = app->minion->enqueue('essn',
    [
      series => '6h',
      run_id => $run_id,
      v2 => $jobs->{essn_v2},
    ],
    {
      expire => 18 * 60,
    },
  );

  my @pred_times = pred_times($run_time);

  for my $holdout_time (@holdout_times) {
    push @pred_times, $holdout_time unless grep { $_ == $holdout_time} @pred_times;
  }

  my @preds;
  for my $station (@stations) {
    my $pred = app->minion->enqueue('pred',
      [
        run_id => $run_id,
        target => [ @pred_times ],
        station => $station,
        ($jobs->{new_kernel} ? (kernels => 'new') : ()),
      ],
      {
        parents => [ $essn_24h ],
        attempts => 2,
        queue => 'pred',
        expire => 18 * 60,
      },
    );
    push @preds, $pred;
  }

  my @html_deps;
  my @holdout_deps;

  for my $render (@target_times) {
    my $irimap = app->minion->enqueue('irimap',
      [
        run_id => $run_id,
        target => $render->{target_time},
        series => '24h',
      ],
      {
        parents => [ $essn_24h ],
        attempts => 2,
        expire => 18 * 60,
      },
    );

    my $ipe;
    if ($jobs->{ipe}) {
      my $ipe_state = $state->{ipe};
      my $target = $render->{target_time};
      if (defined (my $prev = $state->{ipe}{$target})) {
        $ipe = app->minion->enqueue('copy_ipe',
          [
            run_id => $run_id,
            from_run_id => $prev->{run_id},
            target => $target,
          ],
          {
            parents => [ $prev->{job_id} ],
            attempts => 2,
          },
        );
      } else {
        $ipe = app->minion->enqueue('ipe',
          [
            run_id => $run_id,
            target => $render->{target_time},
          ],
          {
            attempts => 2,
          },
        );
        $state->{ipe}{$target} = {
          run_id => $run_id,
          job_id => $ipe,
        };
      }

      if (0 && $jobs->{make_maps}) {
        for my $metric (qw(mufd fof2)) {
          my $map = app->minion->enqueue('rendersvg',
            [
              run_id => $run_id,
              target => $render->{target_time},
              metric => "${metric}_ipe",
              name   => $render->{name},
              format => 'normal',
              dots   => 'none',
              file_format => ['svg'],
            ],
            {
              parents => [ $ipe ],
              attempts => 2,
              expire => 18 * 60,
            }
          );
          push @html_deps, $map;
        }
      }
    }

    my $assimilate = app->minion->enqueue('assimilate',
      [
        run_id => $run_id,
        target => $render->{target_time},
        holdout => ($jobs->{holdout_all_timestep} ? 1 : 0),
        ($jobs->{basemap_type} ? (basemap => $jobs->{basemap_type}) : ()),
      ],
      {
        parents => [ @preds, $irimap, ($jobs->{ipe} ? $ipe : ()) ],
        attempts => 2,
        expire => 18 * 60,
        queue => 'assimilate',
      },
    );
    if ($jobs->{make_maps}) {
      my @map_jobs = make_maps(
        run_id => $run_id,
        target => $render->{target_time},
        name => $render->{name},
        dots => $render->{dots},
        parents => [ $assimilate ],
      );
      push @html_deps, @map_jobs;
      push @holdout_deps, @map_jobs;
    } else {
      push @html_deps, $assimilate;
      push @holdout_deps, $assimilate;
    }


    # This is inside of the loop because of its dependence on the assimilate
    # for the same target time.
    if ($jobs->{band_quality} && $render->{target_time} == $first_target_time) {
      my $band_quality = app->minion->enqueue('band_quality',
        [
          run_id => $run_id,
          target => $render->{target_time},
        ],
        {
          parents => [ $assimilate ],
          attempts => 2,
        },
      );
      push @html_deps, $band_quality;
    }
  }

  my @finish_deps;
  if ($jobs->{renderhtml}) {
    my $renderhtml = app->minion->enqueue('renderhtml',
      [
        run_id => $run_id,
        ($experiment ? (run_name => $experiment) : ()),
      ],
      {
        parents => [ @html_deps ],
        expire => 18 * 60,
        attempts => 2,
      },
    );
    @finish_deps = $renderhtml;
  } else {
    @finish_deps = @html_deps;
  }

  for my $holdout_time (@holdout_times) {
    my $irimap = app->minion->enqueue('irimap',
      [
        run_id => $run_id,
        target => $holdout_time,
        series => '24h',
      ],
      {
        parents => [ $essn_24h ],
        attempts => 2,
        expire => 18 * 60,
      },
    );
    my $assimilate = app->minion->enqueue('assimilate',
      [
        run_id => $run_id,
        target => $holdout_time,
        holdout => 1,
      ],
      {
        parents => [ @preds, $irimap ],
        attempts => 2,
        expire => 18 * 60,
        queue => 'assimilate',
      },
    );
    push @holdout_deps, $assimilate;
  }

  if (@$holdouts) {
    my $holdout_eval = app->minion->enqueue('holdout_evaluate',
      [
        run_id => $run_id,
      ],
      {
        parents => [ @holdout_deps ],
        attempts => 2,
        expire => 3 * 60 * 60,
      },
    );
  }

  app->minion->enqueue('finish_run',
    [
      run_id => $run_id,
    ],
    {
      parents => [ @finish_deps ],
      attempts => 2,
      expire => 3 * 60 * 60,
    },
  );
}

sub fallback_run {
  my ($run_time, $last_data) = @_;
  my @target_times = target_times($run_time);
  my $first_target_time = $target_times[0]{target_time};

  my @stations = app->pg->db->query("select distinct(station_id) from measurement where time > now() - interval '14 day' and time <= now()")->hashes->map(sub { $_->{station_id} })->each;

  my $essn_24h = app->minion->enqueue('essn',
    [
      series => '24h',
      ts => $last_data,
    ],
    {
      attempts => 2,
      expire => 18 * 60,
    },
  );

  my $run_id = $essn_24h;
  app->pg->db->query('insert into runs (id, started, target_time, experiment, state) values (?, to_timestamp(?), to_timestamp(?), ?, ?)',
    $run_id, time(), $first_target_time, 'stale_data', 'created'
  );

  my $essn_6h = app->minion->enqueue('essn',
    [
      series => '6h',
      run_id => $run_id,
      ts => $last_data,
    ]
  );

  my @pred_times = pred_times($run_time);

  my @preds;
  for my $station (@stations) {
    my $pred = app->minion->enqueue('pred',
      [
        run_id => $run_id,
        target => [ @pred_times ],
        station => $station,
      ],
      {
        parents => [ $essn_24h ],
        attempts => 2,
        queue => 'pred',
        expire => 18 * 60,
      },
    );
    push @preds, $pred;
  }

  my @html_deps;

  for my $render (@target_times) {
    my $irimap = app->minion->enqueue('irimap',
      [
        run_id => $run_id,
        target => $render->{target_time},
        series => '24h',
      ],
      {
        parents => [ $essn_24h ],
        attempts => 2,
        expire => 18 * 60,
      },
    );

    my $assimilate = app->minion->enqueue('assimilate',
      [
        run_id => $run_id,
        target => $render->{target_time},
      ],
      {
        parents => [ @preds, $irimap ],
        attempts => 2,
        expire => 18 * 60,
        queue => 'assimilate',
      },
    );

    my @map_jobs = make_maps(
      run_id => $run_id,
      target => $render->{target_time},
      name => $render->{name},
      dots => 'none',
      parents => [ $assimilate ],
    );
    push @html_deps, @map_jobs;
  }

  my $renderhtml = app->minion->enqueue('renderhtml',
    [
      run_id => $run_id,
      last_data => $last_data,
      fallback_banner => 1,
    ],
    {
      parents => [ @html_deps ],
      expire => 18 * 60,
      attempts => 2,
    },
  );

  app->minion->enqueue('finish_run',
    [
      run_id => $run_id,
    ],
    {
      parents => [ $renderhtml ],
      attempts => 2,
      expire => 3 * 60 * 60,
    },
  );
}



sub queue_job {
  my ($run_time, $resched) = @_;

  my $last_data = app->pg->db->query("select extract(epoch from min(last_meas) + interval '1 hour') from (select max(time) as last_meas from measurement group by station_id order by last_meas desc limit 8)")->array->[0];
  if ($last_data < $run_time - 3*3600) {
    fallback_run($run_time, 0 + $last_data);
    goto RESCHED;
  }

  my $num_holdouts = 1;

  my $holdout_meas = $num_holdouts && eval {
    Mojo::UserAgent->new->inactivity_timeout(30)->post("http://localhost:$ENV{API_PORT}/holdout_measurements", form => { num => $num_holdouts })->result->json
  } || [];

  my $state = {
      holdout_meas => $holdout_meas,
  };

  one_run($run_time, $state, undef, {
    no_holdout => 1,
    make_maps => 1,
    renderhtml => 1,
    band_quality => 1,
  });

  my @experiments = (
    sub {
      one_run($run_time, $state, '2024-03-ipe-control', {
          no_holdout => 1,
      });
    },
    sub {
      one_run($run_time, $state, '2024-03-ipe-ipe', {
          no_holdout => 1,
          ipe => 1,
          make_maps => 1,
          renderhtml => 1,
          basemap_type => 'ipe',
      });
    },
    sub {
      one_run($run_time, $state, '2024-03-ipe-blend', {
          no_holdout => 1,
          ipe => 1,
          make_maps => 1,
          renderhtml => 1,
          basemap_type => 'iri-ipe',
      });
    },
    sub {
      one_run($run_time, $state, '2024-03-ipe-linscale', {
          no_holdout => 1,
          ipe => 1,
          make_maps => 1,
          renderhtml => 1,
          basemap_type => 'ipe_scaled',
      });
    },
    sub {
      one_run($run_time, $state, '2024-03-ipe-linscale-blend', {
          no_holdout => 1,
          ipe => 1,
          make_maps => 1,
          renderhtml => 1,
          basemap_type => 'iri-ipe_scaled',
      });
    },
    sub {
      one_run($run_time, $state, '2024-03-ipe-logscale', {
          no_holdout => 1,
          ipe => 1,
          make_maps => 1,
          renderhtml => 1,
          basemap_type => 'ipe_logscaled',
      });
    },
    sub {
      one_run($run_time, $state, '2024-03-ipe-logscale-blend', {
          no_holdout => 1,
          ipe => 1,
          make_maps => 1,
          renderhtml => 1,
          basemap_type => 'iri-ipe_logscaled',
      });
    },
  );

  $_->() for shuffle @experiments;

  app->minion->enqueue('cleanup');

  RESCHED: if ($resched) {
    my ($next, $wait) = next_run();
    Mojo::IOLoop->timer($wait => sub { queue_job($next, 1) });
  }
}

# Admin web and periodic job injector
my ($next, $wait) = next_run;
app->log->debug("First run in $wait seconds");
Mojo::IOLoop->timer($wait => sub { queue_job($next, 1) });

get '/run_prev' => sub {
  my $c = shift;
  my ($prev, $next, undef) = prev_next();
  queue_job($prev, 0);
  $c->render(text => "OK\n");
};

get '/run_next' => sub {
  my $c = shift;
  my ($prev, $next, undef) = prev_next();
  queue_job($next, 0);
  $c->render(text => "OK\n");
};

get '/run_now' => sub {
  my $c = shift;
  queue_job(time, 0);
  $c->render(text => "OK\n");
};

get '/cleanup_now' => sub {
  my $c = shift;
  app->minion->enqueue('cleanup');
  $c->render(text => "OK\n");
};

app->start;
