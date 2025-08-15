# Sample antenna data for testing
our @ANTENNA_TYPES = (
    { key => 'ISOTROPIC', name => 'Isotropic' },
    { key => 'DIPOLE', name => 'Dipole' },
    { key => 'YAGI', name => 'Yagi' },
);

our %ANTENNA_META = (
    'ISOTROPIC' => { gos => 1 },
    'DIPOLE' => { height => 1 },
    'YAGI' => { height => 1, azimuth => 1 },
);

our @ANTENNA_HEIGHTS = (
    { key => '35ft', name => '35 feet' },
    { key => '50ft', name => '50 feet' },
    { key => '0.25wl', name => '0.25 wavelengths' },
);

1;