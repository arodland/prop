package Task::eSSN;
use Mojo::Base 'Mojolicious::Plugin';
use Mojo::UserAgent;

sub register {
  my ($self, $app) = @_;

  $app->minion->add_task(essn => sub {
      my ($job, %args) = @_;
      my $run_id = $args{run_id} || $job->id;

      my $res = Mojo::UserAgent->new->inactivity_timeout(300)->post("http://localhost:$ENV{ESSN_PORT}/generate" =>
        form => {
          series => $args{series},
          run_id => $run_id,
          holdouts => $args{holdouts},
        },
      )->result;
      $res->is_success or die $res->error;
    });
}

1;
