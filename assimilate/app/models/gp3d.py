import numpy as np
import george
from george.kernels import Matern32Kernel

def sph_to_xyz(lat, lon):
    lon = lon * np.pi / 180.
    lat = lat * np.pi / 180.
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    return x, y, z

class GP3D:
    def __init__(self):
        pass

    def train(self, df, t):
        x, y, z = sph_to_xyz(df['station.latitude'].values, df['station.longitude'].values)
        stdev = 0.237 - 0.170 * df.cs

        kernel = Matern32Kernel(1.0, ndim=3)
        self.gp = george.GP(kernel)
        self.gp.compute(np.column_stack((x,y,z)), stdev + 0.05)
        self.z = t

    def predict(self, latitude, longitude):
        x, y, z = sph_to_xyz(latitude, longitude)
        xyz = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
        pred = self.gp.predict(self.z, xyz, return_cov=False)
        return pred.reshape(x.shape)
