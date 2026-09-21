package Task::Forecast;
use Mojo::Base 'Mojolicious::Plugin';

# One job per run: the anomaly model writes all 25 hourly maps itself. No essn/pred/irimap dependency.
sub register {
  my ($self, $app) = @_;

  $app->minion->add_task(forecast_v2 => sub {
      my ($job, %args) = @_;
      my $res = Mojo::UserAgent->new->inactivity_timeout(600)->post("http://localhost:$ENV{FORECAST_PORT}/forecast_24h", =>
        form => \%args,
      )->result;
      $res->is_success or die $res->error;
    });
}

1;
