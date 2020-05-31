package Task::Render;
use Mojo::Base 'Mojolicious::Plugin';

sub register {
  my ($self, $app) = @_;

  $app->minion->add_task(rendersvg => sub {
      my ($job, %args) = @_;
      my $res = Mojo::UserAgent->new->inactivity_timeout(300)->post('http://prop-renderer:5000/rendersvg', =>
        form => \%args,
      )->result;
      $res->is_success or die $res->error . "\n" . $res->body;
  });

  $app->minion->add_task(renderhtml => sub {
      my ($job, %args) = @_;
      my $res = Mojo::UserAgent->new->inactivity_timeout(300)->post('http://prop-renderer:5000/renderhtml', =>
        form => \%args,
      )->result;
      $res->is_success or die $res->error . "\n" . $res->body;
  });
}

1;
