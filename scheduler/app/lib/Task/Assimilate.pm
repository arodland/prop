package Task::Assimilate;
use Mojo::Base 'Mojolicious::Plugin';

sub register {
  my ($self, $app) = @_;

  $app->minion->add_task(assimilate => sub {
      my ($job, %args) = @_;
      return $job->retry({ delay => (1 + int rand 2) }) unless my $guard = $app->minion->guard('assimilate', 120, {limit => 3});
      my $res = Mojo::UserAgent->new->inactivity_timeout(300)->post("http://localhost:$ENV{ASSIMILATE_PORT}/generate", =>
        form => \%args,
      )->result;
      $res->is_success or die $res->error;
    });
}

1;
