#!/usr/bin/env perl
use strict;
use warnings;
package PropHist;
use DBI;
use Web::Simple;
use JSON::MaybeXS qw(encode_json);
use DateTime;
use DateTime::Format::Strptime;
use feature 'postderef';

my $dbh = DBI->connect(
  "dbi:Pg:dbname=$ENV{DB_NAME};host=$ENV{DB_HOST}",
  $ENV{DB_USER},
  $ENV{DB_PASSWORD},
  {
    RaiseError => 1,
  }
);

my $strp = DateTime::Format::Strptime->new(
  pattern => '%Y-%m-%d %H:%M:%S',
  locale => 'en_US',
  time_zone => 'UTC',
);

sub filter_down {
  my ($target, $arr) = @_;
  my $ratio = $target / @$arr;
  my $accum = 0;
  my @ret;

  for my $elem (@$arr) {
    $accum += $ratio;
    if ($accum >= 1) {
      push @ret, $elem;
      $accum -= 1;
    }
  }
  return @ret;
}

sub dispatch_request {
  'GET + /' => sub {
    [ 200, ['Content-Type' => 'text/plain'], ['hello, world!'] ],
  },
  'GET + /history.json + ?station~&days~' => sub {
    my ($self, $station, $days) = @_;

    $days = 7 unless defined $days;
    my $start_time = DateTime->now(time_zone => "UTC")->subtract(days => $days)->strftime('%Y-%m-%d %H:%M:%S');

    my $sql = 'SELECT * FROM STATION';
    if (defined $station) {
      $sql .= ' WHERE ID = ?';
    }
    $sql .= ' ORDER BY id ASC';

    my $sth = $dbh->prepare($sql);
    $sth->execute((defined($station)?$station:()));
    my $stations = $sth->fetchall_arrayref({});

    my $sth2 = $dbh->prepare(q{
      SELECT time, cs, fof2, mufd, hmf2 from measurement WHERE station_id=? AND time >= ? ORDER BY time ASC
    });

    [ sub {
        my $responder = shift;
        my $writer = $responder->([ 200, ['Content-Type' => 'application/json' ] ]);
        $writer->write("[\n");
        my $first_station = 1;
        for my $station(@$stations) {
          $station->{$_} = 0 + $station->{$_} for qw(latitude longitude);

          my $rows = $sth2->execute($station->{id}, $start_time);

          $writer->write(",\n") unless $first_station;
          $first_station = 0;

          my $json = encode_json($station);
          chop($json); # Remove trailing "}"
          $writer->write($json);
          $writer->write(q{,"history":[});

          my $first_measurement = 1;
          measurement: while (my $measurement = $sth2->fetchrow_arrayref) {
            for (1..4) {
              undef($measurement->[$_]) if defined $measurement->[$_] and $measurement->[$_] eq '';
              next measurement unless defined $measurement->[$_];
              $measurement->[$_] = 0 + $measurement->[$_];
            }

            $writer->write(",") unless $first_measurement;
            $first_measurement = 0;
            $writer->write(encode_json($measurement));
          }
          $writer->write(q(]}));
        };
        $writer->write("\n]\n");
        $writer->close;
      }
    ]
  },
  'GET + /history_v2.json + ?station~&days~&@metrics~' => sub {
    my ($self, $station, $days, $metrics) = @_;
    $days = 7 unless defined $days;
    $metrics = ['fof2', 'mufd', 'hmf2'] unless @$metrics;

    my $start_time = DateTime->now(time_zone => 'UTC')->subtract(days => $days)->strftime('%Y-%m-%d %H:%M:%S');

    my @metrics_quoted = map $dbh->quote_identifier($_), @$metrics;
    my $metrics_quoted = join ', ', @metrics_quoted;

    my $sql = 'SELECT * FROM STATION';
    if (defined $station) {
      $sql .= ' WHERE ID = ?';
    }
    $sql .= ' ORDER BY id ASC';

    my $sth = $dbh->prepare($sql);
    $sth->execute((defined($station)?$station:()));
    my $stations = $sth->fetchall_arrayref({});

    my $sth2 = $dbh->prepare(qq{
      SELECT EXTRACT(epoch FROM time), cs, $metrics_quoted FROM measurement
      WHERE station_id=? AND time >= ?
      ORDER BY time ASC
    });

    [ sub {
        my $responder = shift;
        my $writer = $responder->([ 200, [ 'Content-Type' => 'application/json' ] ]);
        $writer->write("[\n");
        my $first_station = 1;
        for my $station (@$stations) {
          $station->{$_} = 0 + $station->{$_} for qw(latitude longitude);
          my $rows = $sth2->execute($station->{id}, $start_time);
          next if $rows == 0;
          $writer->write(",\n") unless $first_station;
          $first_station = 0;
          my $json = encode_json($station);
          chop($json); # Remove trailing "}"
          $writer->write($json);
          $writer->write(q{,"history":[});

          my $first_measurement = 1;
          measurement: while(my $measurement = $sth2->fetchrow_arrayref) {
            my $nonnull = 0;
            for my $i (0 .. $#$measurement) {
              $measurement->[$i] = undef if defined($measurement->[$i]) and $measurement->[$i] eq '';
              $measurement->[$i] = 0 + $measurement->[$i] if defined $measurement->[$i];
              $nonnull = 1 if $i > 1 and defined $measurement->[$i];
            }
            next measurement unless $nonnull;

            $writer->write(",") unless $first_measurement;
            $first_measurement = 0;
            $writer->write(encode_json($measurement));
          }
          $writer->write(q(]}));
        };
        $writer->write("\n]\n");
        $writer->close;
      }
    ]
  },
  'GET + /sample.json' => sub {
    my ($self) = @_;

    my $sql = 'SELECT * FROM STATION WHERE id=(SELECT station_id FROM measurement ORDER BY random() LIMIT 1)';
    my $sth = $dbh->prepare($sql);
    $sth->execute();
    my $stations = $sth->fetchall_arrayref({});

    my $sth2 = $dbh->prepare(q{
      SELECT time FROM measurement WHERE station_id=? ORDER BY random() LIMIT 1
    });

    my $sth3 = $dbh->prepare(q{
      SELECT time, cs, fof2, mufd, hmf2 from measurement WHERE station_id=? AND time BETWEEN (?::timestamp - interval '4 days') AND (?::timestamp + interval '4 days') ORDER BY time ASC
    });

    [ sub {
        my $responder = shift;
        my $writer = $responder->([ 200, ['Content-Type' => 'application/json' ] ]);
        $writer->write("[\n");
        my $first_station = 1;
        for my $station(@$stations) {
          my ($start_time) = $dbh->selectrow_array($sth2, {}, $station->{id});

          $station->{$_} = 0 + $station->{$_} for qw(latitude longitude);

          my $rows = $sth3->execute($station->{id}, $start_time, $start_time);
          next if $rows == 0;

          $writer->write(",\n") unless $first_station;
          $first_station = 0;

          my $json = encode_json($station);
          chop($json); # Remove trailing "}"
          $writer->write($json);
          $writer->write(q{,"history":[});

          my $first_measurement = 1;
          measurement: while (my $measurement = $sth3->fetchrow_arrayref) {
            for (1..4) {
              undef($measurement->[$_]) if defined($measurement->[$_]) and $measurement->[$_] eq '';
              next measurement unless defined $measurement->[$_];
              $measurement->[$_] = 0 + $measurement->[$_];
            }

            $writer->write(",") unless $first_measurement;
            $first_measurement = 0;
            $writer->write(encode_json($measurement));
          };
          $writer->write(q(]}));
        };
        $writer->write("\n]\n");
        $writer->close;
      }
    ]
  },
  'GET + /mixscale.json + ?station~&points~&max_span~' => sub {
    my ($self, $station_id, $max_points, $max_span) = @_;
    $max_points = 2000 unless defined $max_points;

    my $sql = ( defined($station_id) 
      ? 'SELECT * FROM STATION WHERE id=?'
      : 'SELECT * FROM STATION WHERE id=(SELECT station_id FROM measurement ORDER BY random() LIMIT 1)'
    );

    my $sth = $dbh->prepare($sql);
    $sth->execute((defined($station_id) ? ($station_id) : ()));
    my $stations = $sth->fetchall_arrayref({});

    my $sth2 = $dbh->prepare(q{
      SELECT time, cs, fof2, mufd, hmf2 from measurement WHERE station_id=? ORDER BY time ASC
    });

    for my $station(@$stations) {
      $station->{$_} = 0 + $station->{$_} for qw(latitude longitude);
      my $measurements = $dbh->selectall_arrayref($sth2, {}, $station->{id});
      my $ts = [ map { $strp->parse_datetime($_->[0])->epoch } @$measurements ];
      if (defined $max_span) {
        my $secs = $max_span * 86400;
        while ($ts->[-1] - $ts->[0] > $secs) {
          shift @$ts;
          shift @$measurements;
        }
      }

      my $span = $ts->[-1] - $ts->[0];

      while (@$measurements > $max_points) {
        my $idx = int rand @$measurements;
        my $prob = ($ts->[-1] - $ts->[$idx]) / $span + 0.05;
        if (rand(1) < $prob) {
          splice @$measurements, $idx, 1, ();
          splice @$ts, $idx, 1, ();
        }
      }

      for my $measurement (@$measurements) {
        for (1..4) {
          undef($measurement->[$_]) if defined $measurement->[$_] and $measurement->[$_] eq '';
          $measurement->[$_] = 0 + $measurement->[$_] if defined $measurement->[$_];
        }
      }

      @$measurements = grep { defined($_->[1]) && defined($_->[2]) && defined($_->[3]) && defined($_->[4]) } @$measurements;

      $station->{history} = $measurements;
    }

    [ 200, ['Content-Type' => 'application/json'], [encode_json($stations)] ];
  },
  'GET + /mixscale_metric.json + ?station~&points~&max_span~&metric=' => sub {
    my ($self, $station_id, $max_points, $max_span, $metric) = @_;
    $max_points = 2000 unless defined $max_points;

    my $sql = ( defined($station_id) 
      ? 'SELECT * FROM STATION WHERE id=?'
      : 'SELECT * FROM STATION WHERE id=(SELECT station_id FROM measurement ORDER BY random() LIMIT 1)'
    );

    my $sth = $dbh->prepare($sql);
    $sth->execute((defined($station_id) ? ($station_id) : ()));
    my $stations = $sth->fetchall_arrayref({});

    my $metric_quoted = $dbh->quote_identifier($metric);
    my $sth2 = $dbh->prepare(qq{
      SELECT time, cs, $metric_quoted from measurement WHERE station_id=? AND $metric_quoted IS NOT NULL ORDER BY time ASC
    });

    for my $station(@$stations) {
      $station->{$_} = 0 + $station->{$_} for qw(latitude longitude);
      my $measurements = $dbh->selectall_arrayref($sth2, {}, $station->{id});
      my $ts = [ map { $strp->parse_datetime($_->[0])->epoch } @$measurements ];
      if (defined $max_span) {
        my $secs = $max_span * 86400;
        while ($ts->[-1] - $ts->[0] > $secs) {
          shift @$ts;
          shift @$measurements;
        }
      }

      my $span = $ts->[-1] - $ts->[0];

      while (@$measurements > $max_points) {
        my $idx = int rand @$measurements;
        my $prob = ($ts->[-1] - $ts->[$idx]) / $span + 0.05;
        if (rand(1) < $prob) {
          splice @$measurements, $idx, 1, ();
          splice @$ts, $idx, 1, ();
        }
      }

      for my $measurement (@$measurements) {
        for (1..2) {
          undef($measurement->[$_]) if defined $measurement->[$_] and $measurement->[$_] eq '';
          $measurement->[$_] = 0 + $measurement->[$_] if defined $measurement->[$_];
        }
      }

      @$measurements = grep { defined($_->[1]) && defined($_->[2]) } @$measurements;

      $station->{history} = $measurements;
    }

    [ 200, ['Content-Type' => 'application/json'], [encode_json($stations)] ];
  },
  'GET + /4d_history.json + ?metric=&min_cs~&max_points~&max_span~' => sub {
      my ($self, $metric, $min_cs, $max_points, $max_span) = @_;
      if ($metric !~ /^(?:fof2|hmf2|mufd|md)$/) {
          return [400, ['Content-Type' => 'text/plain'], ['invalid metric']];
      }

      $min_cs = 25 unless defined $min_cs;
      $max_points = 4000 unless defined $max_points;
      $max_points = 0 + int($max_points);
      $max_span = 7 unless defined $max_span;

      my $start_time = DateTime->now(time_zone => "UTC")->subtract(days => $max_span)->strftime('%Y-%m-%d %H:%M:%S');

      my ($meas_query, $null_query);
      if ($metric eq 'md') {
          $meas_query = 'm.mufd / m.fof2';
          $null_query = 'm.mufd is not null and m.fof2 is not null';
      } else {
          $meas_query = "m.$metric";
          $null_query = "m.$metric is not null";
      }

      my $sql = qq{select m.station_id, extract(epoch from m.time) as time, $meas_query as meas, m.cs, s.latitude, s.longitude
      from measurement m
      join station s on m.station_id=s.id
      where m.time >= ?
      and $null_query
      and (m.cs >= ? or m.cs = -1)
      and s.use_for_maps=true
      order by m.time asc
      };
      my $sth = $dbh->prepare($sql);
      $sth->execute($start_time, $min_cs);
      my %by_station;
      my $total = 0;
      while (my $row = $sth->fetchrow_arrayref) {
        my ($station_id, @rest) = @$row;
        push $by_station{$station_id}->@*, \@rest;
        $total++;
      }

      # Sort from fewest entries to most
      my @stations = sort { $by_station{$a}->@* <=> $by_station{$b}->@* } keys %by_station;

      my @rows;
      while (@stations) {
        my $s = shift @stations;
        my $size = $by_station{$s}->@*;
        my $target = $max_points / (1 + @stations);
        if ($size <= $target) {
          push @rows, (delete $by_station{$s})->@*;
          $total -= $size;
          $max_points -= $size;
          warn "Station $s: accept all $size\n";
        } else {
          my @filtered = filter_down($target, delete $by_station{$s});
          my $accepted = @filtered;
          push @rows, @filtered;
          $total -= $size;
          $max_points -= $accepted;
          warn "Station $s: accept $accepted out of $size\n";
        }
      }

      [ sub {
          my $responder = shift;
          my $writer = $responder->([ 200, ['Content-Type' => 'application/json' ] ]);
          $writer->write("[\n");
          my $first_row = 1;
          for my $row (@rows) {
            $writer->write(",\n") unless $first_row;
            $first_row = 0;
            $_ = 0+$_ for @$row;
            $row->[0] = int $row->[0];
            $writer->write(encode_json($row));
          }
          $writer->write("\n]\n");
        $writer->close;
      }
    ]
  },
}

PropHist->run_if_script;
