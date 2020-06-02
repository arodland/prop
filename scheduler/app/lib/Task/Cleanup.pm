package Task::Cleanup;
use Mojo::Base 'Mojolicious::Plugin';
use Path::Tiny;

sub register {
  my ($self, $app) = @_;

  $app->minion->add_task(cleanup => sub {
      my ($job, %args) = @_;
      my $db = $app->pg->db;

      my $runs = $db->query(q{select id from runs where state in ('created', 'finished') and started < now() - interval '24 hours' order by started asc limit 10});

      while (my $run = $runs->hash) {
        eval {
          archive_run($db, $run->{id});
        };
        if ($@) {
          $app->log->warn("$@ archiving run $run->{id}");
        }
      }

      $runs = $db->query(q{select id from runs where state='archived' and started < now() - interval '6 months' order by started asc limit 10});

      while (my $run = $runs->hash) {
        eval {
          delete_run($db, $run->{id});
        };
        if ($@) {
          $app->log->warn("$@ deleting run $run->{id}");
        }
      }
  });
}

sub archive_run {
  my ($db, $run_id) = @_;

  my $archive_dir = path("/archive/$run_id");
  $archive_dir->mkpath;

  # essn: leave alone
  
  # pred: leave alone

  # irimap: download from db, place in /archive, delete from db
  my $maps = $db->query("select extract(epoch from time) as ts, dataset from irimap where run_id=?", $run_id);
  while (my $map = $maps->hash) {
    my $map_dir = $archive_dir->child("irimap");
    $map_dir->mkpath;
    my $target_file = $map_dir->child("$map->{ts}.h5");
    $target_file->spew_raw($map->{dataset});
  }
  $db->query("delete from irimap where run_id=?", $run_id);

  # assimilated: download from db, place in /archive, delete from db
  $maps = $db->query("select extract(epoch from time) as ts, dataset from assimilated where run_id=?", $run_id);
  while (my $map = $maps->hash) {
    my $map_dir = $archive_dir->child("assimilated");
    $map_dir->mkpath;
    my $target_file = $map_dir->child("$map->{ts}.h5");
    $target_file->spew_raw($map->{dataset});
  }
  $db->query("delete from assimilated where run_id=?", $run_id);
  
  # rendered files: move to /archive
  my $rendered_dir = path("/output/$run_id");
  if ($rendered_dir->exists) {
    my $dest_dir = $archive_dir->child("rendered");
    $dest_dir->mkpath;

    for my $file ($rendered_dir->children) {
      $file->copy($dest_dir);
    }
    $rendered_dir->remove_tree;
  }
  
  # run: update state to 'archived'
  $db->query("update runs set state=? where id=?", "archived", $run_id);
}

sub delete_run {
  my ($db, $run_id) = @_;
  my $archive_dir = path("/archive/$run_id");

  # essn: leave alone (we can afford this, or manually delete...)

  # pred: delete
  $db->query("delete from prediction where run_id=?", $run_id);
  
  # irimap, assimilated, rendered files: delete from /archive
  $archive_dir->remove_tree if $archive_dir->exists;

  # run: update state to 'deleted'
  $db->query("update runs set state=? where id=?", "deleted", $run_id);
}

1;
