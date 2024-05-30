import numpy as np

def sph_to_xyz(lat, lon):
    lon = lon * np.pi / 180.
    lat = lat * np.pi / 180.
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    return x, y, z

def gen_coords(lat, lon, tm):
    uthour = (tm % 86400) / 3600
    xyz_ef = sph_to_xyz(lat, lon)
    xyz_sf = sph_to_xyz(lat, lon + 15. * uthour)

    # [0]: timestamp, 1 unit = 1 day
    # [1:4]: XYZ in an earth-fixed frame
    # [4:7]: XYZ in a sun-fixed frame
    return [ tm / 86400., *xyz_ef, *xyz_sf ]
