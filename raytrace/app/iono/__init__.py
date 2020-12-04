import numpy as np
from numba import jit
import h5py
import urllib.request
import io
import math

@jit('f8[:](f8[:,:],f8[:],f8[:])',nopython=True,error_model='numpy')
def bilerp(table, latitude, longitude):
    lat = latitude * 180 / np.pi + 90
    lon = longitude * 180 / np.pi + 180
    out = np.zeros_like(lat)

    for i in range(lat.size):
        latc = math.ceil(lat[i])
        lonf = math.floor(lon[i])
        lonc = math.ceil(lon[i])
        latf, latc = int(np.floor(lat[i])), int(np.ceil(lat[i]))
        latp = lat[i] - latf
        lonf, lonc = int(np.floor(lon[i])), int(np.ceil(lon[i]))
        lonp = lon[i] - lonf

        top = lonp * table[latc, lonc] + (1.-lonp) * table[latc, lonf]
        bot = lonp * table[latf, lonc] + (1.-lonp) * table[latf, lonf]

        out[i] = latp * top + (1.-latp) * bot

    return out

class Spline:
    def __init__(self, table):
        self.table = np.array(table)

    def predict(self, latitude, longitude):
        return bilerp(self.table, latitude.flatten(), longitude.flatten()).reshape(longitude.shape)

class Iono:
    def __init__(self, url):
        with urllib.request.urlopen(url) as res:
            content = res.read()
            bio = io.BytesIO(content)
            self.h5 = h5py.File(bio, 'r')
            self.hmf2 = self.spline('/maps/hmf2')
            self.fof2 = self.spline('/maps/fof2')
            self.foe = self.spline('/maps/foe')
            self.gyf = self.spline('/maps/gyf')

    def spline(self, ds):
        return Spline(self.h5[ds])

