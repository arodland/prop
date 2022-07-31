package Task::Pred;
use Mojo::Base 'Mojolicious::Plugin';

sub register {
  my ($self, $app) = @_;

  $app->minion->add_task(pred => sub {
      my ($job, %args) = @_;
      my $res = Mojo::UserAgent->new->inactivity_timeout(300)->post("http://localhost:$ENV{PRED_PORT}/generate" =>
        form => \%args
      )->result;
      $res->is_success or die $res->error;
    });

  $app->minion->add_task(pred_v2 => sub {
      my ($job, %args) = @_;
      my $res = Mojo::UserAgent->new->inactivity_timeout(300)->post("http://localhost:$ENV{PRED_PORT}/generate_v2" =>
        form => \%args
      )->result;
      $res->is_success or die $res->error;
    });
}

1;
