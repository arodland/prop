#!/usr/bin/perl
use strict;
use warnings;

use Data::SAO4;
use Data::Dumper;
use DBI;
use Time::HiRes 'time';
use Net::Statsd::Client;

my $dbh = DBI->connect(
  "dbi:Pg:dbname=$ENV{DB_NAME};host=$ENV{DB_HOST}",
  $ENV{DB_USER},
  $ENV{DB_PASSWORD},
  {
    RaiseError => 1,
  }
);

my $statsd = Net::Statsd::Client->new(
  host => $ENV{STATSD_HOST},
);

warn "Input file: ", $ARGV[0], "\n";

my $sao = Data::SAO4->new(filename => $ARGV[0]);

my $code = $sao->station_code;
my $name = $sao->name || $code;
my $geophys = $sao->geophysical_constants;
my ($lat, $lon) = ($geophys->{latitude}, $geophys->{longitude});

sub debug {
  warn "[$code] ", @_, "\n";
}

my ($station_id) = $dbh->selectrow_array("SELECT id FROM station WHERE code=?", undef, $code);
if (!defined $station_id) {
  ($station_id) = $dbh->selectrow_array("INSERT INTO station (name, code, longitude, latitude, giro, use_for_essn) VALUES (?, ?, ?, ?, FALSE, FALSE) RETURNING id", undef, $name, $code, $lon, $lat);
  debug "Created station ID $station_id";
} else {
  debug "Found station ID $station_id";
}

my $ts = $sao->timestamp;
my $time = $ts->{date} . " " . $ts->{time};
debug "Timestamp: $time";

my ($measurement_id) = $dbh->selectrow_array("SELECT id FROM measurement WHERE station_id=? AND time=?", undef, $station_id, $time);
if ($measurement_id) {
  debug "Already loaded";
  exit 0;
}

if ($ts->{epoch} > time() + 3600) {
  debug "Timestamp is in the future, something isn't right.";
  exit 1;
}

my @cols = ('station_id', 'time', 'cs', 'source');
my @vals = ($station_id, $time, 0+$sao->confidence, 'noaa');
my @placeholders = ('?', '?', '?', '?');

my %map = (
  fof2 => 'foF2',
  fof1 => 'foF1',
  mufd => 'MUF(D)',
  md => 'M(D)',
  foes => 'foEs',
  foe => 'foE',
  hf2 => q{h'F2},
  he => q{h'E},
  hme => 'zmE',
  hmf2 => 'zmF2',
  hmf1 => 'zmF1',
  yf2 => 'yF2',
  yf1 => 'yF1',
  tec => 'TEC',
  scalef2 => 'scaleF2',
  fbes => 'fbEs',
);

my $characteristics = $sao->scaled_characteristics;

# These stations are sending M(D) mislabeled as MUF(D)
if (($code =~ /^(?:MM168|SD266|KB548|MG560|TK356)$/) && 
    defined $characteristics->{'MUF(D)'} && $characteristics->{'MUF(D)'} <= 4.2 && 
    !defined $characteristics->{'M(D)'}) {
  $characteristics->{'M(D)'} = delete $characteristics->{'MUF(D)'};
}

# Compute mufd from fof2 and md or fof2 and hmf2, if available
# (pred can only work with stations that have fof2 + hmf2 + mufd, so it's worth trying to fill in)
if (defined $characteristics->{'foF2'} && defined $characteristics->{'M(D)'} && !defined $characteristics->{'MUF(D)'}) {
  $characteristics->{'MUF(D)'} = $characteristics->{'foF2'} * $characteristics->{'M(D)'};
} elsif (defined $characteristics->{'foF2'} && defined $characteristics->{'zmF2'} && !defined $characteristics->{'MUF(D)'}) {
  $characteristics->{'M(D)'} = 1 / cos(atan2(sin(1500/6371), 1 + ($characteristics->{'zmF2'}+37)/6371 - cos(1500/6371)));
  $characteristics->{'MUF(D)'} = $characteristics->{'foF2'} * $characteristics->{'M(D)'};
}

for my $key (sort keys %map) {
  if (defined(my $val = $characteristics->{$map{$key}})) {
    next if $val == 0;
    push @cols, $key;
    push @vals, $val;
    push @placeholders, '?';
  }
}

my $sql = "INSERT INTO measurement (". join(", ", @cols) .") VALUES (". join(", ", @placeholders) .")";

$dbh->do($sql, undef, @vals);

my $latency = sprintf "%.2f", (time() - $ts->{epoch});

$statsd->increment('prop.noaa_loader.loaded.total');
$statsd->increment("prop.noaa_loader.loaded.station.$code");
$statsd->timing_ms("prop.noaa_loader.latency.overall", $latency * 1000);
$statsd->timing_ms("prop.noaa_loader.latency.station.$code", $latency * 1000);

debug "Latency: $latency";
