package Task::IRIMap;
use Mojo::Base 'Mojolicious::Plugin';
use Mojo::UserAgent;

sub register {
  my ($self, $app) = @_;

  $app->minion->add_task(irimap => sub {
      my ($job, %args) = @_;
      #      return $job->retry({ delay => 10 })
      #        unless my $guard = $app->minion->guard('irimap', 120);

      my $res = Mojo::UserAgent->new->inactivity_timeout(300)->post('http://irimap:5000/generate' =>
        form => \%args
      )->result;
      $res->is_success or die $res->error;
    });
}

1;
