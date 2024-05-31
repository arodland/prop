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
