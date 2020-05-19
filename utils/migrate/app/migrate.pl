#!/usr/bin/perl
use strict;
use warnings;

use DBI;

my $giro_db = DBI->connect(
  "dbi:Pg:dbname=giro;host=$ENV{DB_HOST}",
  $ENV{DB_USER},
  $ENV{DB_PASSWORD},
  { RaiseError => 1 },
);

my $prop_db = DBI->connect(
  "dbi:Pg:dbname=prop;host=$ENV{DB_HOST}",
  $ENV{DB_USER},
  $ENV{DB_PASSWORD},
  { RaiseError => 1 },
);

my $sth = $giro_db->prepare("select * from station order by id asc");
$sth->execute;

while (my $station = $sth->fetchrow_hashref) {
  $prop_db->do("insert into station (id, name, code, longitude, latitude, giro, use_for_essn) values (?, ?, ?, ?, ?, ?, ?)",
    undef,
    $station->{id}, $station->{name}, $station->{code}, $station->{longitude}, $station->{latitude},
    (defined($station->{active}) ? 0 : 1),
    1,
  );
}

$sth = $giro_db->prepare("select * from measurement order by time asc");
$sth->execute;

my $n = 0;

$prop_db->begin_work;
while (my $meas = $sth->fetchrow_hashref) {
  delete $meas->{id};
  my $alt = delete $meas->{altitude};
  my $source = defined($alt) ? 'giro' : 'noaa';

  my @keys = keys %$meas;
  my @vals = @$meas{@keys};
  for my $val (@vals) {
    if (defined($val) && $val eq "") {
      $val = undef;
    }
  }

  my $sql = "insert into measurement(source, " . join(',', @keys) . ") values (?," . join(',', ('?')x@keys) . ")";
  $prop_db->do($sql, undef, $source, @vals);
  if ((++$n) % 10000 == 0) {
    print "$n\n";
    $prop_db->commit;
    $prop_db->begin_work;
  }
}

$prop_db->commit;
