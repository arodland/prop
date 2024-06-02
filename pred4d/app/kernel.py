from tinygp import kernels
import jax.numpy as jnp

class TimeDistance(kernels.stationary.Distance):
    def distance(self, x1, x2):
        if jnp.shape(x1) != (10,) or jnp.shape(x2) != (10,):
            raise ValueError("TimeDistance is defined for 10-vector")
        return jnp.abs(x1[0] - x2[0])

class EarthFrameDistance(kernels.stationary.Distance):
    def distance(self, x1, x2):
        if jnp.shape(x1) != (10,) or jnp.shape(x2) != (10,):
            raise ValueError("EarthFrameDistance is defined for 10-vector")
        return jnp.linalg.norm(x1[1:4] - x2[1:4])

class SolarMagDistance(kernels.stationary.Distance):
    def distance(self, x1, x2):
        if jnp.shape(x1) != (10,) or jnp.shape(x2) != (10,):
            raise ValueError("EarthFrameDistance is defined for 10-vector")
        return jnp.linalg.norm(x1[4:7] - x2[4:7])

class SolarFrameDistance(kernels.stationary.Distance):
    def distance(self, x1, x2):
        if jnp.shape(x1) != (10,) or jnp.shape(x2) != (10,):
            raise ValueError("EarthFrameDistance is defined for 10-vector")
        return jnp.linalg.norm(x1[7:10] - x2[7:10])

def make_4d_kernel(p):
    td = TimeDistance()
    efd = EarthFrameDistance()
    smd = SolarMagDistance()
    sfd = SolarFrameDistance()

    emk = jnp.exp(p[0]) * kernels.Exp(jnp.exp(p[1]), distance=td) * kernels.Matern32(jnp.exp(p[2]), distance=efd)
    smk = jnp.exp(p[3]) * kernels.Exp(jnp.exp(p[4]), distance=td) * kernels.Matern32(jnp.exp(p[5]), distance=smd)
    sfk = jnp.exp(p[6]) * kernels.Exp(jnp.exp(p[7]), distance=td) * kernels.Matern32(jnp.exp(p[8]), distance=sfd)
    return emk + smk + sfk


metric_params = {
    'fof2': [0.28373806900286713, 0.95, 96.75631805227141, 64.57881547797192, -3.8993283155365317, -1.5258040633854937, -1.8951486191566902, -4.124500814042155, 14.207694321741958, -1.087397137773361, -4.035001597241746, 4.286889250545919, -0.6462805249356496],
    'hmf2': [0.27782866288485675, 0.95, 90.39779639145657, 63.325526105039756, -3.9206481628580216, -1.420926047901087, -1.6519746852039672, -0.8920766893132804, 12.980324038511293, 0.031171035138313197, 1.3362516959428512, 8.999935409646529, 1.1229029134866024],
    'mufd': [0.2851248010127453, 0.95, 91.78751021110371, 61.18299291604731, -4.023878303209026, -1.3722807717551049, -1.7068591622304248, -2.4224572845222205, 15.598103667065459, -0.29659021091549626, -0.8960596475440286, 6.330833775866543, 0.6494965365288847],
    'md': [0.25249521163066985, 0.95, 92.6330508077136, 55.90271261413475, -3.8506835690947367, -1.4426857928562495, -1.7058781876553235, -1.7675473433610305, 14.211628300681737, -0.25764046794398976, -1.0167781390963395, 6.875917906429389, 0.34666909311680594],
}
