import numpy as np
import scipy.interpolate
import h5py
import urllib.request
import io

class Spline:
    def __init__(self, table):
        self.table = table
        lat = np.linspace(-np.pi / 2, np.pi / 2, 181)
        lon = np.linspace(-np.pi, np.pi, 361)

        self.spline = scipy.interpolate.RectBivariateSpline(lat, lon, self.table)

    def predict(self, latitude, longitude):
        return self.spline(latitude.flatten(), longitude.flatten(), grid=False).reshape(longitude.shape)

class Iono:
    def __init__(self, url):
        with urllib.request.urlopen(url) as res:
            content = res.read()
            bio = io.BytesIO(content)
            self.h5 = h5py.File(bio, 'r')
            self.hmf2 = self.spline('/maps/hmf2')
            self.fof2 = self.spline('/maps/fof2')
            self.foe = self.spline('/maps/foe')

    def spline(self, ds):
        return Spline(self.h5[ds])

