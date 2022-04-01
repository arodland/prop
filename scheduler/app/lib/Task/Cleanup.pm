package Task::Cleanup;
use Mojo::Base 'Mojolicious::Plugin';
use Path::Tiny;
use Mojolicious::Types;
use Mojo::UserAgent;
use Mojo::URL;
use Mojo::Util qw(b64_encode b64_decode);
use WWW::Google::Cloud::Auth::ServiceAccount;

has 'auth' => sub {
  WWW::Google::Cloud::Auth::ServiceAccount->new(
    credentials_path => '/archiver.json',
  )
};

has 'types' => sub {
  Mojolicious::Types->new
};

has 'base_url' => sub {
  Mojo::URL->new("https://storage.googleapis.com/upload/storage/v1/b/$ENV{ARCHIVE_BUCKET}/o/")
};

sub gcs_upload {
  my ($self, $minion, %args) = @_;
  $args{data} = b64_encode($args{data}) if defined $args{data};
  $minion->enqueue('gcs_upload', [ %args ], {
      attempts => 3,
      queue => 'gcs_upload',
  });
}

sub register {
  my ($self, $app) = @_;

  $app->minion->add_task(cleanup => sub {
      my ($job, %args) = @_;
      my $db = $app->pg->db;

      my $runs = $db->query(q{select id from runs where state='archived' and started < now() - interval '14 days' order by started asc limit 50});

      while (my $run = $runs->hash) {
        eval {
          $self->delete_run($app->minion, $db, $run->{id});
        };
        if ($@) {
          $app->log->warn("$@ deleting run $run->{id}");
        }
      }

      $runs = $db->query(q{select id from runs where state in ('created', 'finished') and started < now() - interval '7 days' order by started asc limit 10});

      while (my $run = $runs->hash) {
        eval {
          $self->archive_run($app->minion, $db, $run->{id});
        };
        if ($@) {
          $app->log->warn("$@ archiving run $run->{id}");
        }
      }

  });

  $app->minion->add_task(gcs_upload => sub {
    my ($job, %args) = @_;
    my $token = $self->auth->get_token;
    my $ua = Mojo::UserAgent->new;
    my $mime_type = $self->types->file_type($args{name});
    my $url = $self->base_url->clone;
    $args{name} =~ s{^/}{};

    my $data;
    if (defined $args{disk_file}) {
      $data = path($args{disk_file})->slurp_raw;
    } elsif (defined $args{data}) {
      $data = b64_decode($args{data});
    } else {
      die "no data found";
    }

    $url->query({
        uploadType => 'media',
        name => "prop-archive/$args{name}",
      });

    my $res = $ua->post(
      $url,
      {
        'Content-Type' => $mime_type || 'application/octet-stream',
        'Authorization' => "Bearer $token",
      },
      $data,
    )->result;

    $job->fail($res->to_string) unless $res->is_success;
  });
}

sub archive_run {
  my ($self, $minion, $db, $run_id) = @_;

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
    $self->gcs_upload($minion, name => "/$run_id/irimap/$map->{ts}.h5", disk_file => "$target_file");
  }
  $db->query("delete from irimap where run_id=?", $run_id);

  # assimilated: download from db, place in /archive, delete from db
  $maps = $db->query("select extract(epoch from time) as ts, dataset from assimilated where run_id=?", $run_id);
  while (my $map = $maps->hash) {
    my $map_dir = $archive_dir->child("assimilated");
    $map_dir->mkpath;
    my $target_file = $map_dir->child("$map->{ts}.h5");
    $target_file->spew_raw($map->{dataset});
    $self->gcs_upload($minion, name => "/$run_id/assimilated/$map->{ts}.h5", disk_file => "$target_file");
  }
  $db->query("delete from assimilated where run_id=?", $run_id);
  
  # rendered files: move to /archive
  my $rendered_dir = path("/output/$run_id");
  if ($rendered_dir->exists) {
    my $dest_dir = $archive_dir->child("rendered");
    $dest_dir->mkpath;

    for my $file ($rendered_dir->children) {
      next unless "$file" =~ /-(?:now|6h|12h|24h)(?:\.|_station)/;

      my $target_file = $file->copy($dest_dir);
      $self->gcs_upload($minion, name => "/$run_id/rendered/" . $file->basename, disk_file => "$target_file");
    }
    $rendered_dir->remove_tree;
  }

  # run: update state to 'archived'
  $db->query("update runs set state=? where id=?", "archived", $run_id);
}

sub delete_run {
  my ($self, $minion, $db, $run_id) = @_;
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
