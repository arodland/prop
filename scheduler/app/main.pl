#!/usr/bin/perl
use Mojolicious::Lite;
use Mojo::IOLoop;
use Mojo::Pg;
use Mojo::UserAgent;

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
plugin 'Task::Assimilate';
plugin 'Task::BandQuality';
plugin 'Task::Render';
plugin 'Task::Cleanup';
plugin 'Task::HoldoutEvaluate';

sub next_run {
  my $INTERVAL = 900; # 15 minutes
  my $LEAD = 300; # Run 5 minutes early, e.g. :10, :25, :40, :55

  my $now = time;
  my $prev = $now - (($now + $LEAD) % $INTERVAL);
  my $next = $prev + $INTERVAL;
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

  return @jobs;
}


sub queue_job {
  my ($run_time, $resched) = @_;

  my $num_holdouts = 1;

  my $holdouts = eval { 
    Mojo::UserAgent->new->inactivity_timeout(30)->post("http://localhost:$ENV{API_PORT}/holdout", form => { num => $num_holdouts })->result->json 
  } || [];

  my @holdout_ids = map $_->{holdout}, @$holdouts;
  my @holdout_times = map $_->{ts}, @$holdouts;

  my $essn_24h = app->minion->enqueue('essn',
    [
      series => '24h',
      holdouts => [ @holdout_ids ],
    ],
    {
      attempts => 2,
      expire => 18 * 60,
    },
  );

  my $run_id = $essn_24h;
  app->pg->db->query('insert into runs (id, started, state) values (?, to_timestamp(?), ?)',
    $run_id, time(), 'created'
  );

  my $essn_6h = app->minion->enqueue('essn',
    [
      series => '6h',
    ],
    {
      expire => 18 * 60,
    },
  );

  my @target_times = target_times($run_time);
  my @pred_times = pred_times($run_time);

  for my $holdout_time (@holdout_times) {
    push @pred_times, $holdout_time unless grep { $_ == $holdout_time} @pred_times;
  }

  my $pred = app->minion->enqueue('pred',
    [
      run_id => $run_id,
      target => [ @pred_times ],
    ],
    {
      parents => [ $essn_24h ],
      attempts => 2,
      queue => 'pred',
      expire => 18 * 60,
    },
  );

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
        parents => [ $pred, $irimap ],
        attempts => 2,
        expire => 18 * 60,
        queue => 'assimilate',
      },
    );
    my @map_jobs = make_maps(
      run_id => $run_id,
      target => $render->{target_time},
      name => $render->{name},
      dots => $render->{dots},
      parents => [ $assimilate ],
    );

    push @html_deps, @map_jobs;

    if ($render->{target_time} == $target_times[0]{target_time}) {
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

  my $renderhtml = app->minion->enqueue('renderhtml',
    [
      run_id => $run_id,
    ],
    {
      parents => [ @html_deps ],
      expire => 18 * 60,
      attempts => 2,
    },
  );

  my @holdout_deps;
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
        parents => [ $pred, $irimap ],
        attempts => 2,
        expire => 18 * 60,
        queue => 'assimilate',
      },
    );
    push @holdout_deps, $assimilate;
  }

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

  app->minion->enqueue('cleanup');

  if ($resched) {
    my ($next, $wait) = next_run();
    Mojo::IOLoop->timer($wait => sub { queue_job($next, 1) });
  }
}

# Admin web and periodic job injector
my ($next, $wait) = next_run;
app->log->debug("First run in $wait seconds");
Mojo::IOLoop->timer($wait => sub { queue_job($next, 1) });

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
