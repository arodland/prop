import wmm2020 as wmm
import numpy as np
from scipy.interpolate import RectBivariateSpline

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

def modip_compute(lat, lon, dt):
    year_start = dt.replace(month=1, day=1)
    year_end = year_start.replace(year=year_start.year + 1)
    year_dec = dt.year + ((dt - year_start).total_seconds()) / float((year_end - year_start).total_seconds())

    mag = wmm.wmm(lat, lon, 300, year_dec)
    modip = np.degrees(np.arctan2(np.radians(np.array(mag.incl)), np.sqrt(np.cos(np.radians(lat)))))

    return modip

class ModipSpline:
    def __init__(self, dt):
        lat = np.linspace(-90, 90, 181)
        lon = np.linspace(-180, 180, 361)

        latm, lonm = np.meshgrid(lat, lon, indexing='ij')
        modip = modip_compute(latm, lonm, dt)
        self.spline = RectBivariateSpline(lat, lon, modip)

    def modip(self, lat, lon, _):
        latx, lonx = np.atleast_1d(lat, lon)
        val = self.spline(latx.flatten(), lonx.flatten(), grid=False)
        if np.ndim(lat) == 0:
            return val.item()
        else:
            return val.reshape(lat.shape)

def gen_coords_pluggable(lat, lon, tm, dt, modip_func):
    uthour = (tm % 86400) / 3600
    modip = modip_func(lat, lon, dt)

    xyz_em = sph_to_xyz(modip, lon)
    xyz_sm = sph_to_xyz(modip, lon + 15. * uthour)
    xyz_sf = sph_to_xyz(lat, lon + 15. * uthour)

    return [np.ones_like(lat) * tm / 86400., *xyz_em, *xyz_sm, *xyz_sf ]
