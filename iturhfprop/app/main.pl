#!/usr/bin/perl
use Mojolicious::Lite -signatures, -async_await;
use Mojo::File;
use Mojo::UserAgent;
use Mojo::IOLoop::ReadWriteFork;
use Ham::Locator;

my @TARGETS = (
    [ 'UA Moscow', '55.76, 37.62' ],
    [ 'UA Yakutsk, Siberia', '62.04, 129.74' ],
    [ 'JA Tokyo', '35.68, 139.65' ],
    [ '9V Singapore', '1.352, 103.82' ],
    [ 'VU Hyderabad', '17.39, 78.49' ],
    [ 'F Paris', '48.86, 2.35' ],
    [ 'LA Trondheim', '63.34, 10.40' ],
    [ 'I Rome', '41.90, 12.50' ],
    [ '4X Tel Aviv', '32.08, 34.78' ],
    [ 'ZL Wellington', '-41.29, 174.78' ],
    [ 'VK Perth', '-31.95, 115.86' ],
    [ 'VK Sydney', '-33.86, 151.21' ],
    [ 'KH6 Honolulu', '21.31, -157.86' ],
    [ '5W Western Samoa', '-13.76, -172.11' ],
    [ '3B8 Mauritius', '-20.35, 57.55' ],
    [ 'ZS Johannesburg', '-26.20, 28.05' ],
    [ '5Z Nairobi', '-1.29, 36.82' ],
    [ 'EA8 Canary Isles', '28.29, -16.63' ],
    [ 'LU Buenos Aires', '-34.63, -58.38' ],
    [ 'PY Rio de Janiero', '-22.91, -43.17' ],
    [ 'OA Lima', '-12.05, -77.04' ],
    [ 'W New Orleans', '29.95, -90.07' ],
    [ 'W Washington DC', '38.90, -77.04' ],
    [ 'VE Quebec', '46.82, -71.21' ],
    [ 'KL Anchorage', '61.22, -149.9' ],
    [ 'VE Vancouver', '49.28, -121.12' ],
    [ 'W San Francisco', '37.77, -122.42' ],
);

my @BANDS = (
    ['10m', 28.850, 'contest'],
    ['12m', 24.940, 'non-contest'],
    ['15m', 21.225, 'contest'],
    ['17m', 18.118, 'non-contest'],
    ['20m', 14.175, 'contest'],
    ['30m', 10.120, 'non-contest'],
    ['40m', 7.150, 'contest'],
    ['60m', 5.330, 'non-contest'],
    ['80m', 3.650, 'contest'],
    ['160m', 1.900, 'contest'],
);

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


async sub one_run {
    my ($c, %params) = @_;

    my ($txlat, $txlon) = parse_locator($params{txloc});
    my ($rxlat, $rxlon) = parse_locator($params{rxloc});

    my $data_dir = Mojo::File::tempdir;

    my $mt = Mojo::Template->new->escape(sub {
            my ($in) = @_;
            $in =~ s/"//g;
            $in =~ s/\n//g;
            return qq{"$in"};
        })->vars(1);

    my $input = $mt->render_file(
        $c->app->home->child('templates', 'p2p.ep'), {
            year => (gmtime)[5]+1900,
            month => (gmtime)[4]+1,
            ssn => $params{run_info}{essn},
            txant => $c->param('txant'),
            txpow => 10 * log($c->param('txpow') / 1000) / log(10),
            txlat => $txlat,
            txlon => $txlon,
            rxlat => $rxlat,
            rxlon => $rxlon,
            data_dir => $data_dir->to_string . "/",
            freqs => [ map $_->[1], @BANDS ],
        });

    my $input_file = $data_dir->child('p2p.txt')->spew($input, 'UTF-8');
    my $output_file = Mojo::File::tempfile(DIR => $data_dir);

    my $template_dir = Mojo::File::path('/opt/iturhfprop/data');

    for my $child ($template_dir->list->each) {
        if ($child->basename =~ /^ionos\d+\.bin$/i) {
            symlink($params{iono_bin}, $data_dir->child($child->basename));
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
    # system("ITURHFProp", $input_file, $output_file);

    my $fh = $output_file->open('<');
    my (@table, %freq2row, $seen_header);
    for my $line (<$fh>) {
        chomp $line;
        last if $line =~ /\*End Calculated Parameters/;
        $seen_header = 1 and next if $line =~ /\* Calculated Parameters/;
        next unless $line =~ /\S/;
        next unless $seen_header;
        my @fields = split /\s*,\s*/, $line;
        my (undef, $hour, $freq, $muf50, $muf90, $muf10, $pr, $bcr) = @fields;
        my $pop = prop_prob($freq, $muf90, $muf50, $muf10);

        if (!defined $freq2row{$freq}) {
            $freq2row{$freq} = @table;
        }
        my $row = $freq2row{$freq};
        $table[$row][$hour-1] = {
            color => color($bcr),
            smeter => smeter($pr),
            pr => $pr,
            bcr => $bcr,
            pop => $pop,
        };
        if ($pr < -151 || $bcr < 10 || $pop < 10) {
            $table[$row][$hour-1]{blank} = Mojo::JSON->true;
            $table[$row][$hour-1]{color} = "bcr-0";
        }
    };

    return \@table;
}

get '/radcom' => sub ($c) {
    $c->stash(targets => \@TARGETS);
};

post '/radcom' => async sub ($c) {
    my $ua = Mojo::UserAgent->new;
    my $hl = Ham::Locator->new;
    my $api_url = Mojo::URL->new("http://localhost/")->port($ENV{API_PORT});

    my $run_info = $ua->get(Mojo::URL->new('/latest_hourly.json')->to_abs($api_url))->result->json;
    $run_info->{hour} = (gmtime($run_info->{maps}[0]{ts}))[2];

    my $iono_bin = Mojo::File::tempfile;
    my $res = $ua->get(Mojo::URL->new('/iongrid.bin')->to_abs($api_url)->query({ run_id => $run_info->{run_id} }));
    $res->result->save_to($iono_bin->to_string);

    my $out = "";
    my @results;

    my $tx = $c->render_later->tx;

    for my $target (@TARGETS) {
        my ($name, $rxloc) = @$target;
        push @results, {
            name => $name,
            table => await one_run(
                $c,
                %{ $c->req->params->to_hash },
                rxloc => $rxloc,
                iono_bin => $iono_bin,
                run_info => $run_info,
            ),
        };
    }
    $c->stash(results => \@results);
    $c->stash(bands => \@BANDS);
    $c->stash(run_info => $run_info);
    $c->render(template => 'radcom_result');
};

get '/radcom_table' => async sub($c) {
    my $ua = Mojo::UserAgent->new;
    my $hl = Ham::Locator->new;
    my $api_url = Mojo::URL->new("http://localhost/")->port($ENV{API_PORT});

    my $run_info = $ua->get(Mojo::URL->new('/latest_hourly.json')->to_abs($api_url))->result->json;
    $run_info->{hour} = (gmtime($run_info->{maps}[0]{ts}))[2];

    my $iono_bin = Mojo::File::tempfile;
    $c->stash(iono_bin => $iono_bin);
    my $res = $ua->get(Mojo::URL->new('/iongrid.bin')->to_abs($api_url)->query({ run_id => $run_info->{run_id} }));
    $res->result->save_to($iono_bin->to_string);

    my $tx = $c->render_later->tx;

    my $table = await one_run(
        $c,
        %{ $c->req->params->to_hash },
        iono_bin => $iono_bin,
        run_info => $run_info,
    );

    $c->stash(table => $table);
    $c->stash(bands => \@BANDS);
    $c->stash(run_info => $run_info);
    $c->stash(start_hour => $c->param('start_hour'));
    $c->render(template => 'radcom_table');
};

get '/radcom.json' => async sub ($c) {
    my $ua = Mojo::UserAgent->new;
    my $hl = Ham::Locator->new;
    my $api_url = Mojo::URL->new("http://localhost/")->port($ENV{API_PORT});

    my $run_info = $ua->get(Mojo::URL->new('/latest_hourly.json')->to_abs($api_url))->result->json;
    $run_info->{hour} = (gmtime($run_info->{maps}[0]{ts}))[2];

    my $iono_bin = Mojo::File::tempfile;
    my $res = $ua->get(Mojo::URL->new('/iongrid.bin')->to_abs($api_url)->query({ run_id => $run_info->{run_id} }));
    $res->result->save_to($iono_bin->to_string);

    my $tx = $c->render_later->tx;

    my $table = await one_run(
        $c,
        %{ $c->req->params->to_hash },
        iono_bin => $iono_bin,
        run_info => $run_info,
    );

    $c->render(json => {
            table => $table,
            bands => \@BANDS,
    });
};

app->hook('before_dispatch' => sub {
    my $self = shift;
    $self->req->url->base->path('/hfprop/');
});

app->start;

