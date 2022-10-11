package Data::SAO4;

use Moo;
use Fortran::Format;
use POSIX qw(mktime);

has 'filename' => (is => 'ro');
has 'fh' => (is => 'lazy');
sub _build_fh {
  my ($self) = @_;
  open my $fh, '<', $self->filename or die $!;
  return $fh;
}

has 'fix_ionfm' => (is => 'ro', default => 1);

has 'index' => (is => 'rwp');
has 'data' => (is => 'rwp');
has 'by_shortname' => (is => 'rwp');
has 'by_longname' => (is => 'rwp');

# MONKEY SEE, MONKEY PATCH
sub Fortran::Format::Edit::E::read_once {
  my ($self) = @_;
  return undef unless $self->writer->want_more;
  my $s = $self->writer->read($self->{width});
  $self->writer->put(0 + $s);
}

my @GROUPS = (
  [ 'Geophysical Constants', 'geophysical_constants', '16F7.3' ],
  [ 'System Description and Operator\'s Message', 'desc', 'A120' ],
  [ 'Time Stamp and Sounder Settings', 'timestamp_settings', '120A1' ],
  [ 'Scaled Ionospheric Characteristics', 'scaled_characteristics', '15F8.3' ],
  [ 'Analysis Flags', 'analysis_flags', '60I2' ],
  [ 'Doppler Translation Table', 'doppler_table', '16F7.3' ],
  # F2 O
  [ 'F2 O Virtual Heights', 'f2o_vh', '15F8.3' ],
  [ 'F2 O True Heights', 'f2o_th', '15F8.3' ],
  [ 'F2 O Amplitudes', 'f2o_a', '40I3' ],
  [ 'F2 O Doppler Numbers', 'f2o_d', '120I1' ],
  [ 'F2 O Frequencies', 'f2o_f', '15F8.3' ],
  # F1 O
  [ 'F1 O Virtual Heights', 'f1o_vh', '15F8.3' ],
  [ 'F1 O True Heights', 'f1o_th', '15F8.3' ],
  [ 'F1 O Amplitudes', 'f1o_a', '40I3' ],
  [ 'F1 O Doppler Numbers', 'f1o_d', '120I1' ],
  [ 'F1 O Frequencies', 'f1o_f', '15F8.3' ],
  # E O
  [ 'E O Virtual Heights', 'eo_vh', '15F8.3' ],
  [ 'E O True Heights', 'eo_th', '15F8.3' ],
  [ 'E O Amplitudes', 'eo_a', '40I3' ],
  [ 'E O Doppler Numbers', 'eo_d', '120I1' ],
  [ 'E O Frequencies', 'eo_f', '15F8.3' ],
  # F2 X
  [ 'F2 X Virtual Heights', 'f2x_vh', '15F8.3' ],
  [ 'F2 X Amplitudes', 'f2x_a', '40I3' ],
  [ 'F2 X Doppler Numbers', 'f2x_d', '120I1' ],
  [ 'F2 X Frequencies', 'f2x_f', '15F8.3' ],
  # F1 X
  [ 'F1 X Virtual Heights', 'f1x_vh', '15F8.3' ],
  [ 'F1 X Amplitudes', 'f1x_a', '40I3' ],
  [ 'F1 X Doppler Numbers', 'f1x_d', '120I1' ],
  [ 'F1 X Frequencies', 'f1x_f', '15F8.3' ],
  # E X
  [ 'E X Virtual Heights', 'ex_vh', '15F8.3' ],
  [ 'E X Amplitudes', 'ex_a', '40I3' ],
  [ 'E X Doppler Numbers', 'ex_d', '120I1' ],
  [ 'E X Frequencies', 'ex_f', '15F8.3' ],
  #
  [ 'Median Amplitudes of F Echoes', 'f_median_amplitude', '40I3' ],
  [ 'Median Amplitudes of E Echoes', 'e_median_amplitude', '40I3' ],
  [ 'Median Amplitudes of Es Echoes', 'es_median_amplitude', '40I3' ],
  [ 'True Heights Coefficients F2 Layer UMLCAR Method', 'f2_true_heights', '10E11.6E1' ],
  [ 'True Heights Coefficients F1 Layer UMLCAR Method', 'f1_true_heights', '10E11.6E1' ],
  [ 'True Heights Coefficients E Layer UMLCAR Method', 'e_true_heights', '10E11.6E1' ],
  [ 'Quasi-Parabolic Segments Fitted to the Profile', 'quasi_parabolic', '6E20.12E2' ],
  [ 'Edit Flags - Characteristics', 'edit_flags', '120I1' ],
  [ 'Valley Description - W,D UMLCAR Model', 'valley_description', '10E11.6E1' ],
  # Es O
  [ 'Es O Virtual Heights', 'eso_vh', '15F8.3' ],
  [ 'Es O Amplitudes', 'eso_a', '40I3' ],
  [ 'Es O Doppler Numbers', 'eso_d', '120I1' ],
  [ 'Es O Frequencies', 'eso_f', '15F8.3' ],
  # E Auroral
  [ 'E Auroral O Virtual Heights', 'ea_vh', '15F8.3' ],
  [ 'E Auroral O Amplitudes', 'ea_a', '40I3' ],
  [ 'E Auroral O Doppler Numbers', 'ea_d', '120I1' ],
  [ 'E Auroral O Frequencies', 'ea_f', '15F8.3' ],
  # True Height Profile
  [ 'True Heights', 'true_heights', '15F8.3'],
  [ 'Plasma Frequencies', 'plasma_frequencies', '15F8.3' ],
  [ 'Electron Densities', 'electron_densities', '15E8.3E1' ],
  # URSI Qualifying and Descriptive Letters
  [ 'Qualifying Letters', 'qualifying_letters', '120A1' ],
  [ 'Descriptive Letters', 'descriptive_letters', '120A1' ],
  [ 'Edit Flags - Traces and Profile', 'edit_flags', '120I1' ],
  # Auroral E Layer Profile Data
  [ 'True Heights Coefficients Ea Layer UMLCAR Method', 'ea_th_umlcar', '10E11.6E1' ],
  [ 'Auroral E True Heights', 'ea_th', '15F8.3' ],
  [ 'Auroral Plasma Frequencies', 'ea_plasma_frequencies', '15F8.3' ],
  [ 'Auroral Electron Densities', 'ea_electron_densities', '15E8.3E1' ],
);

sub BUILD {
  my ($self) = @_;
  $self->parse;
}

sub parse {
  my ($self) = @_;

  (my $idx) = Fortran::Format->new('40I3')->read($self->fh, 80);

  my ($version) = splice @$idx, 79;
  if ($version < 2) {
    warn "Version $version is older than SAO-4, I'm not sure if this will work.";
  } elsif ($version > 5) {
    warn "Version $version is newer than SAO-4.3, unrecognized fields may case parse failure.";
  }

  $self->_set_index($idx);
  
  my ($data, $by_shortname, $by_longname);

  for my $i (0 .. $#$idx) {
    my $n_records = $idx->[$i];
    if ($n_records) {
      my $group = $GROUPS[$i];
      if (!defined $group) {
        die "Records encountered in unknown group ", ($i + 1), ", cannot continue.";
      }
      my ($longname, $shortname, $format) = @$group;
      ($data->[$i]) = Fortran::Format->new($format)->read($self->fh, $n_records);
      $data->[$i] = [ $data->[$i] ] if !ref $data->[$i];
      $by_shortname->{$shortname} = $data->[$i];
      $by_longname->{$longname} = $data->[$i];
    }
  }

  $self->_set_data($data);
  $self->_set_by_shortname($by_shortname);
  $self->_set_by_longname($by_longname);
}

sub geophysical_constants {
  my ($self) = @_;

  my $data = $self->by_shortname->{geophysical_constants};
  return {
    gyrofrequency => $data->[0],
    dip_angle => $data->[1],
    latitude => $data->[2],
    longitude => $data->[3],
    ssn => $data->[4],
  };
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

sub timestamp {
  my ($self) = @_;

  my $data = $self->by_shortname->{timestamp_settings};
  my $ts = join("", @$data[2..18]);
  my ($year, $doy, $month, $day, $hour, $minute, $second) = unpack "A4A3A2A2A2A2A2", $ts;
  my $epoch = timegm($second, $minute, $hour, $day, ($month - 1), ($year - 1900));
  ($second, $minute, $hour, $day, $month, $year) = gmtime($epoch); # normalize
  $month += 1;
  $year += 1900;

  if ($self->fix_ionfm && $self->description =~ /(?:UFOFH|IONFM) CONVERTED TO SAO/) {
    my $year_now = (gmtime)[5] + 1900;
    while ($year < $year_now - 9) {
      $year += 10;
    }
    while ($year > $year_now + 9) {
      $year -= 10;
    }
  }

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

has 'description' => (
  is => 'lazy',
);
sub _build_description {
  my ($self) = @_;
  my $desc = join " ", @{ $self->by_shortname->{desc} };
  $desc =~ s/\r|\n/ /g;
  return $desc;
}

has 'description_firstline' => (
  is => 'lazy',
);
sub _build_description_firstline {
  my ($self) = @_;
  my $desc = $self->by_shortname->{desc}[0];
  $desc =~ s/\s+\z//;
  return $desc;
}

sub station_code {
  my ($self) = @_;

  my @words = split " ", $self->description_firstline;
  if ($words[1] eq '052/,') {
    return 'RL052';
  } elsif ($words[1] eq 'Sodankyla') {
    return 'SO166';
  } elsif ($words[1] =~ m[/(.*),]) {
    return $1;
  } else {
    return;
  }
}

sub confidence {
  my ($self) = @_;

  my $description = $self->description;
  if ($description =~ /Confidence: (\d+)%/) {
    return $1;
  }
  my $flags = $self->by_shortname->{analysis_flags};
  if (defined $flags && $flags->[9]) {
    my ($lower, $upper) = unpack "A1A1", $self->by_shortname->{analysis_flags}->[9];
    my $cl = (($lower > $upper) ? $lower : $upper);
    return ((125 - 25 * $cl) . "e0");
  }
  return -1;
}

sub name {
  my ($self) = @_;

  my $description = $self->description_firstline;
  if ($description =~ /NAME\s+([^,]+)/) {
    return $1;
  }
  return;
}

my @CHARACTERISTICS = (
  "foF2", "foF1", "M(D)", "MUF(D)", "fmin", "foEs", "fminF", "fminE", "foE",
  "fxI", "h'F", "h'F2", "h'E", "h'Es", "zmE", "yE", "QF", "QE", "DownF",
  "DownE", "DownEs", "FF", "FE", "D", "fMUF", "h'(fMUF)", "delta_foF2", "foEp",
  "f(h'F)", "f(h'F2)", "foF1p", "zmF2", "zmF1", "zhalfNm", "foF2p",
  "fminEs", "yF2", "yF1", "TEC", "scaleF2", "B0", "B1", "D1", "foEa", "h'Ea",
  "foP", "h'P", "fbEs", "typeEs",
);

sub scaled_characteristics {
  my ($self) = @_;

  my $params = $self->by_shortname->{scaled_characteristics};
  my $ret = {};

  for my $i (0 .. $#CHARACTERISTICS) {
    last if $i > $#$params;
    next if $params->[$i] == 9999;
    $ret->{$CHARACTERISTICS[$i]} = $params->[$i];
  }

  return $ret;
}

1;
