import wmm2020 as wmm
import numpy as np

def sph_to_xyz(lat, lon):
    lon = lon * np.pi / 180.
    lat = lat * np.pi / 180.
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    return x, y, z

def gen_coords_flat(lat, lon, tm, dt):
    uthour = (tm % 86400) / 3600
    year_start = dt.replace(month=1, day=1)
    year_end = year_start.replace(year=year_start.year + 1)
    year_dec = dt.year + ((dt - year_start).total_seconds()) / float((year_end - year_start).total_seconds())

    mag = wmm.wmm(lat, lon, 300, year_dec)
    modip = np.degrees(np.arctan2(np.radians(np.array(mag.incl.item())), np.sqrt(np.cos(np.radians(lat)))))

    xyz_em = sph_to_xyz(modip, lon)
    xyz_sm = sph_to_xyz(modip, lon + 15. * uthour)
    xyz_sf = sph_to_xyz(lat, lon + 15. * uthour)

    return [tm / 86400., *xyz_em, *xyz_sm, *xyz_sf ]

def gen_coords(lat, lon, tm, dt):
    uthour = (tm % 86400) / 3600
    year_start = dt.replace(month=1, day=1)
    year_end = year_start.replace(year=year_start.year + 1)
    year_dec = dt.year + ((dt - year_start).total_seconds()) / float((year_end - year_start).total_seconds())

    mag = wmm.wmm(lat, lon, 300, year_dec)
    modip = np.degrees(np.arctan2(np.radians(np.array(mag.incl)), np.sqrt(np.cos(np.radians(lat)))))

    xyz_em = sph_to_xyz(modip, lon)
    xyz_sm = sph_to_xyz(modip, lon + 15. * uthour)
    xyz_sf = sph_to_xyz(lat, lon + 15. * uthour)

    return [np.ones_like(lat) * tm / 86400., *xyz_em, *xyz_sm, *xyz_sf ]
