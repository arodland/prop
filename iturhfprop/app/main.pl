#!/usr/bin/perl
use v5.40;
use Mojolicious::Lite -signatures, -async_await;
use Mojo::File;
use Mojo::UserAgent;
use Mojo::IOLoop::ReadWriteFork;
use Ham::Locator;
use FU::Validate;
use FindBin;

require "$FindBin::Bin/data.pl";

plugin 'AccessLog';
app->helper(cache => sub { state $cache = Mojo::Cache->new });

my $API_URL;
if (defined $ENV{KC2G_API}) {
    $API_URL = Mojo::URL->new($ENV{KC2G_API});
} else {
    $API_URL = Mojo::URL->new("http://localhost/")->port($ENV{API_PORT});
}

sub color {
    my ($bcr) = @_;
    my $slice = int($bcr / 10);
    $slice = 0 if $slice < 0;
    $slice = 9 if $slice > 9;
    return "bcr-$slice";
}

sub parse_locator {
    my ($loc) = @_;
    if ($loc =~ /^\s*[\d.+-]+\s*,\s*[\d.+-]+\s*$/) {
        return split /\s*,\s*/, $loc, 2;
    }
    my $hl = Ham::Locator->new;
    $hl->set_loc($loc);
    return $hl->loc2latlng;
}

sub smeter {
    my ($pr) = @_;
    my $relative = $pr + 157;
    my $slice = int($relative / 6);
    $slice = 0 if $slice < 0;
    $slice = 9 if $slice > 9;
    return $slice;
}

# Returns the Probability of Propagation from the upper and lower deciles
# using the method from the following source;
#
# Bradley, P.A. and Bedford, C., 1976. Prediction of HF circuit availability. Electronics Letters, 12(1), pp.32-33.
sub prop_prob($freq, $muf90, $muf50, $muf10) {
    my $prob;
    if ($freq < $muf50) {
        my $f1 = $muf90 / $muf50;
        $prob = 130 - (80 / (1 + ((1-($freq/$muf50))/(1-$f1))));
    } else {
        my $fu = $muf10 / $muf50;
        $prob = (80 / (1 + ((($freq/$muf50)-1)/($fu-1)))) - 30;
    }
    $prob = 100 if $prob > 100;
    $prob = 0 if $prob < 0;
    return $prob;
}

async sub run_iturhfprop($c, $iono_bin, $template, $template_args) {
    my $data_dir = Mojo::File::tempdir;

    my $mt = Mojo::Template->new->escape(sub {
            my ($in) = @_;
            $in =~ s/"//g;
            $in =~ s/\n//g;
            return qq{"$in"};
        })->vars(1);
    my $input = $mt->render_file(
        $template,
        {
            %$template_args,
            data_dir => $data_dir->to_string . "/"
        }
    );
    my $input_file = $data_dir->child('p2p.txt')->spew($input, 'UTF-8');

    my $output_file = Mojo::File::tempfile(DIR => $data_dir);
    my $template_dir = Mojo::File::path('/opt/iturhfprop/data');

    for my $child ($template_dir->list->each) {
        if ($child->basename =~ /^ionos\d+\.bin$/i) {
            symlink($iono_bin, $data_dir->child($child->basename));
        } else {
            symlink($child, $data_dir->child($child->basename));
        }
    }

    my $rwf = Mojo::IOLoop::ReadWriteFork->new;
    $c->stash(rwf => $rwf);
    $rwf->conduit({ type => 'pipe' });
    $rwf->on(read => sub ($rwf, $bytes) {
        STDOUT->syswrite($bytes);
    });
    $rwf->on(error => sub ($rwf, $err) {
        warn $err;
    });
    await $rwf->run_p("/usr/local/bin/ITURHFProp", $input_file, $output_file);
    return $output_file;
}

async sub one_run_p2p($c, %params) {
    my ($txlat, $txlon) = parse_locator($params{txloc});
    my ($rxlat, $rxlon) = parse_locator($params{rxloc});

    my $output_file = await run_iturhfprop(
        $c, $params{iono_bin},
        $c->app->home->child('templates', 'p2p.ep'), {
            year => (gmtime)[5]+1900,
            month => (gmtime)[4]+1,
            ssn => $params{run_info}{essn},
            ($params{txant} eq 'ISOTROPIC'
                ? (txant => 'ISOTROPIC', txgos => $params{txgos})
                : (txant => "/opt/antennas/$params{txant}.mbp", txgos => 0)
            ),
            ($params{rxant} eq 'ISOTROPIC'
                ? (rxant => 'ISOTROPIC', rxgos => $params{rxgos})
                : (rxant => "/opt/antennas/$params{rxant}.mbp", rxgos => 0)
            ),
            txpow => 10 * log($params{txpow} / 1000) / log(10),
            txlat => $txlat,
            txlon => $txlon,
            rxlat => $rxlat,
            rxlon => $rxlon,
            rxnoise => $params{rxnoise},
            snrr => $params{snrr},
            bw => $params{bw},
            freqs => [ map $_->[1], @::BANDS ],
        });

    my $fh = $output_file->open('<');
    my (@table, %freq2row, $seen_header);
    for my $line (<$fh>) {
        chomp $line;
        last if $line =~ /\*End Calculated Parameters/;
        $seen_header = 1, next if $line =~ /\* Calculated Parameters/;
        next unless $line =~ /\S/;
        next unless $seen_header;
        my @fields = split /\s*,\s*/, $line;
        my (undef, $hour, $freq, $muf50, $muf90, $muf10, $pr, $grw, undef, $noise, undef, $bcr) = @fields;
        my $pop = prop_prob($freq, $muf90, $muf50, $muf10);

        $noise = $noise - 204 + 10 * log($params{bw}) / log(10);

        if (!defined $freq2row{$freq}) {
            $freq2row{$freq} = @table;
        }
        my $row = $freq2row{$freq};
        $table[$row][$hour-1] = {
            color => color($bcr),
            smeter => smeter($pr),
            pr => 0+$pr,
            bcr => 0+$bcr,
            noise => 0+$noise,
            noise_s => smeter($noise),
            pop => 0+$pop,
        };
        if ($pr < -151 || $bcr < 10 || $pop < 10) {
            $table[$row][$hour-1]{blank} = Mojo::JSON->true;
            $table[$row][$hour-1]{color} = "bcr-0";
        }
    };

    return \@table;
}

get '/radcom' => sub ($c) {
    $c->res->code(301);
    $c->redirect_to('/planner');
};

get '/radcom_beta' => sub ($c) {
    $c->res->code(301);
    $c->redirect_to('/planner_beta');
};

get '/planner' => sub ($c) {
    $c->stash(
        targets => \@::TARGETS,
        bands => \@::BANDS,
        modes => \@::MODES,
        antenna_types => \@::ANTENNA_TYPES,
        antenna_meta => \%::ANTENNA_META,
        antenna_heights => \@::ANTENNA_HEIGHTS,
    );
};

get '/planner_beta' => sub ($c) {
    $c->stash(
        targets => \@::TARGETS,
        bands => \@::BANDS,
        modes => \@::MODES,
        antenna_types => \@::ANTENNA_TYPES,
        antenna_meta => \%::ANTENNA_META,
        antenna_heights => \@::ANTENNA_HEIGHTS,
    );
};

get '/area' => sub ($c) {
    $c->stash(
        bands => \@::BANDS,
        modes => \@::MODES,
        antenna_types => \@::ANTENNA_TYPES,
        antenna_meta => \%::ANTENNA_META,
        antenna_heights => \@::ANTENNA_HEIGHTS,
    );
};

async sub get_iono_bin($c, $run_id) {
    my $iono_bin = Mojo::File::tempfile;
    my $contents = $c->cache->get("iono_bin;$run_id");
    if (!defined $contents) {
        my $ua = Mojo::UserAgent->new;
        my $res = await $ua->get_p(Mojo::URL->new('iongrid.bin')->to_abs($API_URL)->query({ run_id => $run_id }));
        $contents = $res->result->body;
        $c->cache->set("iono_bin;$run_id" => $contents);
    }
    $iono_bin->spew($contents);
    return $iono_bin;
}

my $locator_pattern = qr/^
    (?:[a-r]{2}\d{2}(?:[a-x]{2}(?:\d{2})?)?
    |-?\d+(?:\.\d+)\s*,\s*-?\d+(?:\.\d+))
$/xi;

my $antenna_pattern = qr/^[a-z0-9_\@.-]+$/i;

my $validation_schema_p2p = {
    type => 'hash',
    unknown => 'remove',
    keys => {
        txloc => { regex => $locator_pattern },
        rxloc => { regex => $locator_pattern },
        txgos => { num => 1, default => 0 },
        rxgos => { num => 1, default => 0 },
        txant => { regex => $antenna_pattern },
        rxant => { regex => $antenna_pattern },
        txpow => { num => 1, min => 0.01 },
        rxnoise => { enum => [ qw(CITY RESIDENTIAL RURAL QUIETRURAL QUIET NOISY) ], default => 'RURAL' },
        snrr => { num => 1, default => 6 },
        bw => { num => 1, default => 3000 },
        start_hour => { enum => [ qw(ZERO ZERO_LOCAL CURRENT) ], default => 'ZERO' },
        tz_offset => { int => 1, min => -24, max => 24, default => 0 },
    },
};
my $validator_p2p = FU::Validate->compile($validation_schema_p2p);

get '/planner_table' => async sub($c) {
    my $tx = $c->render_later->tx;

    my $ua = Mojo::UserAgent->new;
    my $hl = Ham::Locator->new;

    my $run_info;
    if (defined($c->param('run_id'))) {
        $run_info = await $ua->get_p(Mojo::URL->new('run_info.json')->to_abs($API_URL)->query({run_id => $c->param('run_id')}))
            ->then(sub { $_[0]->result->json });
    } else {
        my $url = Mojo::URL->new('latest_hourly.json')->to_abs($API_URL);
        if (defined($c->param('experiment'))) {
            $url->query({ experiment => $c->param('experiment') });
        }
        $run_info = await $ua->get_p($url)
            ->then(sub { $_[0]->result->json });
    }
    $run_info->{hour} = (gmtime($run_info->{maps}[0]{ts}))[2];

    my $iono_bin = await get_iono_bin($c, $run_info->{run_id});

    my $params = $validator_p2p->validate($c->req->params->to_hash);

    my $table = await one_run_p2p(
        $c,
        %$params,
        iono_bin => $iono_bin,
        run_info => $run_info,
    );
    $iono_bin->remove;

    $c->stash(table => $table);
    $c->stash(bands => \@::BANDS);
    $c->stash(run_info => $run_info);
    $c->stash(start_hour => $params->{start_hour});
    $c->stash(tz_offset => $params->{tz_offset});
    $c->res->headers->header('Access-Control-Allow-Origin' => '*');
    $c->render(template => 'planner_table');
};

get '/planner.json' => async sub ($c) {
    my $tx = $c->render_later->tx;

    my $ua = Mojo::UserAgent->new;
    my $hl = Ham::Locator->new;

    my $run_info = await $ua->get_p(Mojo::URL->new('latest_hourly.json')->to_abs($API_URL))
        ->then(sub { $_[0]->result->json });
    $run_info->{hour} = (gmtime($run_info->{maps}[0]{ts}))[2];

    my $iono_bin = await get_iono_bin($c, $run_info->{run_id});

    my $table = await one_run_p2p(
        $c,
        %{ $validator_p2p->validate($c->req->params->to_hash) },
        iono_bin => $iono_bin,
        run_info => $run_info,
    );

    $iono_bin->remove;

    $c->render(json => {
            table => $table,
            bands => \@::BANDS,
    });
};

if (defined $ENV{PATH_PREFIX}) {
    app->hook('before_dispatch' => sub {
        my $self = shift;
        $self->req->url->base->path($ENV{PATH_PREFIX});
    });
}

app->start;

