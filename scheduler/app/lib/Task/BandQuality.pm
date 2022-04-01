package Task::BandQuality;
use Mojo::Base 'Mojolicious::Plugin';

sub register {
  my ($self, $app) = @_;

  $app->minion->add_task(band_quality => sub {
      my ($job, %args) = @_;
      my $res = Mojo::UserAgent->new->inactivity_timeout(300)->post("http://localhost:$ENV{BANDQUALITY_PORT}/generate", =>
        form => \%args,
      )->result;
      $res->is_success or die $res->error;
    });
}

1;
