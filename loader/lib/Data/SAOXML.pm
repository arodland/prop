package Data::SAOXML;

use Moo;
use XML::Twig;
use Encode;
use POSIX qw(mktime);

has 'filename' => (is => 'ro');
has 'station_code' => (is => 'rwp');
has 'name' => (is => 'rwp');
has 'timestamp' => (is => 'rwp');
has 'geophysical_constants' => (is => 'rwp');
has 'scaled_characteristics' => (is => 'rwp');
has 'confidence' => (is => 'rwp');

sub BUILD {
    my ($self) = @_;
    $self->parse;
}

my %URSICodeToName = (
    "00" => "foF2", "01" => "fxF2", "02" => "fzF2", "03" => "M(D)", "04" => "h'F2", "07" => "MUF(D)", "09" => "scaleF2",
    "10" => "foF1", "11" => "fxF1", "14" => "h'F1", "16" => "h'F",
    "20" => "foE", "24" => "h'E",
    "30" => "foEs", "31" => "fxEs", "32" => "fbEs", "34" => "h'Es", "36" => "typeEs",
    "42" => "fmin",
    "60" => "f(h'F2)", "61" => "f(h'F)",
    "70" => "TEC",
    "80" => "fminE", "83" => "yE", "84" => "QF", "85" => "QE", "86" => "FF", "87" => "FE",
    "90" => "zmE", "91" => "zmF1", "92" => "zmF2", "93" => "zhalfNm", "94" => "yF2", "95" => "yF1",
    "D0" => "B0", "D1" => "B1", "D2" => "D1",
);

my %KnownModeled = (
    "foEp" => 1, "foF1p" => 1, "foF2p" => 1,
);

sub parse {
    my ($self) = @_;

    my $twig;
    my $ok = eval {
        $twig = XML::Twig->new;
        $twig->parsefile($self->filename);
        1;
    };
    if (!$ok) {
        # Some files contain latin1 even though they don't have an encoding declaration
        # in the prolog, which means they should be UTF-8. Twig rightly barfs on this.
        # If we see an error that appears to be due to that, try again after converting the file
        # encoding.
        if ($@ =~ /not well-formed/) {
            $twig = XML::Twig->new;
            open my $fh, '<:encoding(latin1)', $self->filename or die "$! opening " . $self->filename;
            my $data = do { local $/; <$fh> };
            $data = Encode::encode('UTF-8', $data);
            $twig->parse($data);
        } else {
            die $@;
        }
    }

    my $rl = $twig->first_elt("SAORecordList");
    die "SAORecordList not found" unless $rl;
    my $sao_record = $rl->first_child("SAORecord");
    die "SAORecord not found" unless $sao_record;

    my $version = $sao_record->att('FormatVersion');
    if ($version > 5) {
        warn "Version $version is newer than SAOXML-5.0, may cause trouble.";
    }

    $self->_set_station_code( $sao_record->att('URSICode') );
    $self->_set_name( $sao_record->att('StationName') );
    $self->_set_timestamp( $self->materialize_timestamp($sao_record->att('StartTimeUTC')) );
    $self->_set_geophysical_constants({
        latitude => $sao_record->att('GeoLatitude'),
        longitude => $sao_record->att('GeoLongitude'),
    });

    $self->_set_confidence( $self->compute_confidence($sao_record) );

    my $chars = $sao_record->first_child('CharacteristicList');
    my @characteristics = $chars ? $chars->children : ();

    my %char_out;
    for my $char (@characteristics) {
        if ($char->tag eq 'URSI') { # URSI standard characteristics, appendix C
            my $charname = $URSICodeToName{$char->att('ID')};
            if (defined $charname) {
                $char_out{$charname} = $char->att('Val');
            }
        } elsif ($char->tag eq 'Modeled') { # ARTIST modeled
            my $charname = $char->att('Name');
            if ($KnownModeled{$charname}) {
                $char_out{$charname} = $char->att('Val');
            }
        }
    }

    $self->_set_scaled_characteristics(\%char_out);
}

sub timegm {
    my ($sec, $min, $hour, $day, $mon, $year) = @_;
    my $epoch = mktime(0, 0, 0, 1, $mon, $year); # First day of month
    $epoch -= $epoch % 86400; # Adjust to UTC midnight
    my @gmtime = gmtime($epoch);
    if ($gmtime[3] > 1) { # Timezone was negative, we went backwards to the last day of the prev month
        $epoch += 86400;
    }
    # Now go forwards to the requested time
    $epoch += 86400 * ($day - 1) + 3600 * $hour + 60 * $min + $sec;
    return $epoch;
}

sub materialize_timestamp {
    my ($self, $ts) = @_;
    
    my @m;
    unless (@m = $ts =~ /^(\d{4})-(\d{2})-(\d{2})(?: -(\d{3}))? (\d{2}):(\d{2}):(\d{2}(?:\.\d*)?)$/) {
        warn "unparseable date $ts";
        return undef;
    }

    my ($year, $month, $day, $doy, $hour, $minute, $second) = @m;
    my $epoch = timegm($second, $minute, $hour, $day, ($month - 1), ($year - 1900));
    ($second, $minute, $hour, $day, $month, $year) = gmtime($epoch); # normalize
    $month += 1;
    $year += 1900;

    return {
        year => $year,
        day_of_year => $doy,
        month => $month,
        day => $day,
        hour => $hour,
        minute => $minute,
        second => $second,
        date => sprintf("%04d-%02d-%02d", $year, $month, $day),
        time => sprintf("%02d:%02d:%02d", $hour, $minute, $second),
        iso => sprintf("%04d-%02d-%02dT%02d:%02d:%02dZ", $year, $month, $day, $hour, $minute, $second),
        epoch => $epoch
    };
}

sub compute_confidence {
    my ($self, $sao_record) = @_;
    my $sys_info = $sao_record->first_child('SystemInfo');
    my $comment = $sys_info ? $sys_info->first_child('Comments') : undef;

    if (defined $comment && $comment->text_only =~ /Confidence: (\d+)%/) {
        return $1;
    }

    my $autoscaler = $sys_info ? $sys_info->first_child('AutoScaler') : undef;
    my $artist_flags = $autoscaler ? $autoscaler->att('ArtistFlags') : undef;

    if (defined $artist_flags) {
        my @flags = split " ", $artist_flags;
        my ($lower, $upper) = unpack "A1A1", $flags[9];
        my $cl = (($lower > $upper) ? $lower : $upper);
        return ((125 - 25 * $cl) . "e0");
    }

    return -1;
}

sub file_format {
    return "saoxml";
}

1;
