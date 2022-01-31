package Minion::Statsd;
use Mojo::Base 'Mojolicious::Plugin';
use Net::Statsd::Client;

my $statsd = Net::Statsd::Client->new(host => $ENV{STATSD_HOST});

sub register {
  my ($self, $app) = @_;

  $app->minion_notifier->on(job => sub {
    my ($notifier, $job_id, $event) = @_;
    return unless $event eq 'finished' or $event eq 'failed';
    my $job = $notifier->minion->job($job_id);
    my $task = $job->task;
    my %info = %{ $job->info };
    my $queue = $info{queue};

    $statsd->increment("minion.job.state.$event");
    $statsd->increment("minion.job.task.$task");
    $statsd->increment("minion.job.task.$task.state.$event");
    $statsd->increment("minion.job.queue.$queue");
    $statsd->increment("minion.job.queue.$queue.state.$event");

    if ($event eq 'finished') {
      my $wait_ms = 1000 * ($info{started} - $info{delayed});
      my $run_ms = 1000 * ($info{finished} - $info{started});
      my $total_ms = 1000 * ($info{finished} - $info{created});

      $statsd->timing_ms("minion.job.wait_time", $wait_ms);
      $statsd->timing_ms("minion.job.run_time", $run_ms);
      $statsd->timing_ms("minion.job.total_time", $total_ms);

      $statsd->timing_ms("minion.job.task.$task.wait_time", $wait_ms);
      $statsd->timing_ms("minion.job.task.$task.run_time", $run_ms);
      $statsd->timing_ms("minion.job.task.$task.total_time", $total_ms);

      $statsd->timing_ms("minion.job.queue.$queue.wait_time", $wait_ms);
      $statsd->timing_ms("minion.job.queue.$queue.run_time", $run_ms);
      $statsd->timing_ms("minion.job.queue.$queue.total_time", $total_ms);
    }
  });
}

1;
