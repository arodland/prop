package Task::Render;
use Mojo::Base 'Mojolicious::Plugin';

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
