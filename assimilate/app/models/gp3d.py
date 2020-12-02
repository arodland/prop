import numpy as np
import rbf
import rbf.gauss

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

        gp = rbf.gauss.gpiso(rbf.basis.mat32, (0.0, 1.0, 1.0))
        self.gp = gp_cond = gp.condition(np.vstack((x,y,z)).T, t, sigma=(stdev + 0.05))

    def predict(self, latitude, longitude):
        x, y, z = sph_to_xyz(latitude, longitude)
        xyz = np.array([x.flatten(), y.flatten(), z.flatten()]).T
        pred, sd = self.gp.meansd(xyz)
        return pred.reshape(x.shape)
