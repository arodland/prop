import numpy as np
from scipy.interpolate import RectBivariateSpline

class Spline:
    def __init__(self, table):
        self.table = table
        lat = np.linspace(-90, 90, 181)
        lon = np.linspace(-180, 180, 361)

        self.spline = RectBivariateSpline(lat, lon, self.table)

    def predict(self, latitude, longitude):
        return self.spline(latitude.flatten(), longitude.flatten(), grid=False).reshape(longitude.shape)
