package Data::SAO;

use Data::SAO4;
use Data::SAOXML;

sub new {
    my ($class, %args) = @_;
    if ($args{filename} =~ /\.xml$/i) {
        return Data::SAOXML->new(%args);
    } else {
        return Data::SAO4->new(%args);
    }
}

1;
