package Task::IRIMap;
use Mojo::Base 'Mojolicious::Plugin';
use Mojo::UserAgent;

sub register {
  my ($self, $app) = @_;

  $app->minion->add_task(irimap => sub {
      my ($job, %args) = @_;
      my $res = Mojo::UserAgent->new->inactivity_timeout(300)->post("http://localhost:$ENV{IRIMAP_PORT}/generate" =>
        form => \%args
      )->result;
      $res->is_success or die $res->error;
    });
}

1;
