#!/usr/bin/env perl
use strict;
use warnings;
package PropHist;
use DBI;
use Web::Simple;
use JSON::MaybeXS qw(encode_json);
use DateTime;
use DateTime::Format::Strptime;

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
          next if $rows == 0;

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
          };
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
      SELECT time, cs, fof2, mufd, hmf2 from measurement WHERE station_id=? AND time BETWEEN (?::timestamp - interval '4 days')::text AND (?::timestamp + interval '4 days')::text ORDER BY time ASC
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
}

PropHist->run_if_script;
