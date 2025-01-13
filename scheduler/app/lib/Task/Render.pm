package Task::Render;
use v5.24;
use Mojo::Base 'Mojolicious::Plugin';
use Path::Tiny;
use DateTime;


our %LOCATIONS = (
  "us-east" => { lat => 37.5, lon => -77.5 },
  "us-central" => { lat => 39, lon => -102 },
  "us-west" => { lat => 41, lon => -121.5 },
  "eu-central" => { lat => 50, lon => 13.5 },
);

sub register {
  my ($self, $app) = @_;

  $app->minion->add_task(rendersvg => sub {
      my ($job, %args) = @_;
      my $res = Mojo::UserAgent->new->inactivity_timeout(300)->post("http://localhost:$ENV{RENDERER_PORT}/rendersvg", =>
        form => \%args,
      )->result;
      $res->is_success or die $res->error . "\n" . $res->body;
  });

  $app->minion->add_task(rendermuf => sub {
      my ($job, $location, %args) = @_;
      my $res = Mojo::UserAgent->new->inactivity_timeout(300)->post("http://localhost:$ENV{RENDERER_PORT}/moflof.svg", =>
        form => { $LOCATIONS{$location}->%*, %args },
      )->result;
      $res->is_success or die $res->error . "\n" . $res->body;
    });

  $app->minion->add_task(renderhtml => sub {
      my ($job, %args) = @_;

      if (delete $args{fallback_banner}) {
        my $dir = path('/output')->child($args{run_id})->mkdir;
        my $banner_file = $dir->child('banner.html');
        my $last_data_ts = DateTime->from_epoch(epoch => $args{last_data})->strftime('%Y-%m-%d %H:%M');
        $banner_file->spew(<<EOHTML);
  <div class="notice">
    <h2>Data is stale</h2>
    <p>
    No data has been received from GIRO for more than 3 hours. Last data was at $last_data_ts UTC.
    Maps are likely to be of low quality.
    </p>
  </div>
EOHTML
        $banner_file->chmod(0644);
      }
      my $res = Mojo::UserAgent->new->inactivity_timeout(300)->post("http://localhost:$ENV{RENDERER_PORT}/renderhtml", =>
        form => \%args,
      )->result;
      $res->is_success or die $res->error . "\n" . $res->body;

  });

  $app->minion->add_task(finish_run => sub {
      my ($job, %args) = @_;
      $app->pg->db->query('update runs set state=?, ended=to_timestamp(?) where id=?',
        'finished', time(), $args{run_id},
      );
  });

}

1;
