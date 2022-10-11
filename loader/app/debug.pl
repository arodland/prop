#!/usr/bin/perl
use strict;
use warnings;

use Data::SAO;
use Data::Dumper;

my $sao = Data::SAO->new(filename => $ARGV[0]);

print "Station code: ", $sao->station_code, "\n";
print "Name: ", $sao->name, "\n";

my $geophys = $sao->geophysical_constants;

print "Lat: $geophys->{latitude} Lon: $geophys->{longitude}\n";
print "Timestamp: ", $sao->timestamp->{date}, " ", $sao->timestamp->{time}, "\n";
print "Confidence: ", (0+$sao->confidence), "\n";

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

for my $key (sort keys %map) {
  if (defined(my $val = $characteristics->{$map{$key}})) {
    print "$key: $val ";
  }
}
print "\n\n";
