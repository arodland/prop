use utf8;
use Mojo::JSON ();

our @TARGETS = (
    [ '5W Western Samoa', '-13.76, -172.11' ],
    [ 'KH6 Honolulu', '21.31, -157.86' ],
    [ 'KL Anchorage', '61.22, -149.9' ],
    [ 'W San Francisco', '37.77, -122.42' ],
    [ 'VE Vancouver', '49.28, -121.12' ],
    [ 'W New Orleans', '29.95, -90.07' ],
    [ 'W Washington DC', '38.90, -77.04' ],
    [ 'OA Lima', '-12.05, -77.04' ],
    [ 'LU Buenos Aires', '-34.63, -58.38' ],
    [ 'VE Quebec', '46.82, -71.21' ],
    [ 'PY Rio de Janeiro', '-22.91, -43.17' ],
    [ 'EA8 Canary Isles', '28.29, -16.63' ],
    [ 'F Paris', '48.86, 2.35' ],
    [ 'LA Trondheim', '63.34, 10.40' ],
    [ 'I Rome', '41.90, 12.50' ],
    [ 'ZS Johannesburg', '-26.20, 28.05' ],
    [ '4X Tel Aviv', '32.08, 34.78' ],
    [ '5Z Nairobi', '-1.29, 36.82' ],
    [ 'UA Moscow', '55.76, 37.62' ],
    [ '3B8 Mauritius', '-20.35, 57.55' ],
    [ 'VU Hyderabad', '17.39, 78.49' ],
    [ '9V Singapore', '1.352, 103.82' ],
    [ 'VK Perth', '-31.95, 115.86' ],
    [ 'UA Yakutsk, Siberia', '62.04, 129.74' ],
    [ 'JA Tokyo', '35.68, 139.65' ],
    [ 'VK Sydney', '-33.86, 151.21' ],
    [ 'ZL Wellington', '-41.29, 174.78' ],
);

our @BANDS = (
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

our @MODES = (
    { name => "WSPR", bw => 2500, snr => -29 },
    { name => "FT8", bw => 50, snr => -3 },
    { name => "CW", bw => 500, snr => 0 } ,
    { name => "SSB (Usable)", bw => 3000, snr => 6 },
    { name => "SSB (Marginal)", bw => 3000, snr => 15 },
    { name => "SSB (Commercial)", bw => 3000, snr => 33 },
    { name => "AM (Fair)", bw => 5000, snr => 36 },
    { name => "AM (Good)", bw => 5000, snr => 48 },
    # { name => "SWBC (Fair)", bw => 5000, snr => 36 },
    # { name => "SWBC (Good)", bw => 5000, snr => 48 },
    # { name => "Voice(1)/600bps Data", bw => 3000, snr => 15 },
    # { name => "Voice(3)/1200bps Data", bw => 3000, snr => 17 },
    # { name => "Voice(5)/2400bps Data", bw => 3000, snr => 19 },
    # { name => "4800bps MIL-110b Data", bw => 3000, snr => 22 },
    # { name => "9600bps MIL-110b Data", bw => 3000, snr => 33 }
);

our @ANTENNA_TYPES = (
    { key => 'ISOTROPIC', name => 'Isotropic' },
    { key => 'dipole', name => 'Dipole' },
    { key => '135doublet', name => "135' Doublet" },
    { key => 'vert-quarter', name => 'Vertical (1/4 WL)' },
    { key => 'vert-fiveeight', name => 'Vertical (5/8 WL)' },
    { key => 'vert-43foot', name => "Vertical (43')" },
    { key => 'yagi-2el', name => 'Yagi (2el)' },
    { key => 'yagi-3el', name => 'Yagi (3el)' },
    { key => 'yagi-4el', name => 'Yagi (4el)' },
    { key => 'yagi-5el', name => 'Yagi (5el)' },
    { key => 'yagi-7el', name => 'Yagi (7el)' },
    { key => 'yagi-9el', name => 'Yagi (9el)' },
    { key => 'yagi-11el', name => 'Yagi (11el)' },
);

our %ANTENNA_META = (
    'ISOTROPIC' => { gos => Mojo::JSON->true },
    'dipole' => { height => Mojo::JSON->true, azimuth => Mojo::JSON->true },
    '135doublet' => { height => Mojo::JSON->true, azimuth => Mojo::JSON->true },
    'yagi-2el' => { height => Mojo::JSON->true, azimuth => Mojo::JSON->true },
    'yagi-3el' => { height => Mojo::JSON->true, azimuth => Mojo::JSON->true },
    'yagi-4el' => { height => Mojo::JSON->true, azimuth => Mojo::JSON->true },
    'yagi-5el' => { height => Mojo::JSON->true, azimuth => Mojo::JSON->true },
    'yagi-7el' => { height => Mojo::JSON->true, azimuth => Mojo::JSON->true },
    'yagi-9el' => { height => Mojo::JSON->true, azimuth => Mojo::JSON->true },
    'yagi-11el' => { height => Mojo::JSON->true, azimuth => Mojo::JSON->true },
);

our @ANTENNA_HEIGHTS = (
    { key => '0.125wl', name => '1/8 wavelength' },
    { key => '0.25wl', name => '1/4 wavelength' },
    { key => '0.375wl', name => '3/8 wavelength' },
    { key => '0.5wl', name => '1/2 wavelength' },
    { key => '0.75wl', name => '3/4 wavelength' },
    { key => '1wl', name => '1 wavelength' },
    { key => '1.5wl', name => '1½ wavelength' },
    { key => '2wl', name => '2 wavelengths' },
    { key => '2.5wl', name => '2½ wavelengths '},
    { key => '10ft', name => '10 feet (3m)' },
    { key => '20ft', name => '20 feet (6m)' },
    { key => '35ft', name => '35 feet (10m)' },
    { key => '50ft', name => '50 feet (15m)' },
    { key => '75ft', name => '75 feet (23m)' },
    { key => '100ft', name => '100 feet (30m)' },
    { key => '125ft', name => '125 feet (38m)' },
    { key => '150ft', name => '150 feet (46m)' },
    { key => '200ft', name => '200 feet (61m)' },
);


1;
