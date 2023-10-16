package Task::IPE;
use Mojo::Base 'Mojolicious::Plugin';
use Mojo::UserAgent;

sub register {
  my ($self, $app) = @_;

  $app->minion->add_task(ipe => sub {
      my ($job, %args) = @_;
      my $ts = $args{target};
      $ts -= ($ts % 300);
      my $key = "ipe-$ts";
      return $job->retry({ delay => (10 + int rand 10) }) unless my $guard = $app->minion->guard($key, 300);
      my $res = Mojo::UserAgent->new->inactivity_timeout(300)->post("http://localhost:$ENV{IPE_PORT}/generate" =>
        form => \%args
      )->result;
      $res->is_success or die $res->error;
    });

    $app->minion->add_task(copy_ipe => sub {
      my ($job, %args) = @_;
      my $db = $app->pg->db;
      my $res = $db->query(
        'INSERT INTO ipemap (time, run_id, dataset) SELECT time, ?, dataset FROM ipemap WHERE run_id=? AND time=to_timestamp(?)', 
        $args{run_id}, $args{from_run_id}, $args{target}
      );
    });
}

1;
