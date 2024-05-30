from tinygp import kernels
import jax.numpy as jnp

class TimeDistance(kernels.stationary.Distance):
    def distance(self, x1, x2):
        if jnp.shape(x1) != (7,) or jnp.shape(x2) != (7,):
            raise ValueError("TimeDistance is defined for 7-vector")
        return jnp.abs(x1[0] - x2[0])

class EarthFrameDistance(kernels.stationary.Distance):
    def distance(self, x1, x2):
        if jnp.shape(x1) != (7,) or jnp.shape(x2) != (7,):
            raise ValueError("EarthFrameDistance is defined for 7-vector")
        return jnp.linalg.norm(x1[1:4] - x2[1:4])

class SolarFrameDistance(kernels.stationary.Distance):
    def distance(self, x1, x2):
        if jnp.shape(x1) != (7,) or jnp.shape(x2) != (7,):
            raise ValueError("EarthFrameDistance is defined for 7-vector")
        return jnp.linalg.norm(x1[4:7] - x2[4:7])

def make_4d_kernel(p):
    td = TimeDistance()
    efd = EarthFrameDistance()
    sfd = SolarFrameDistance()

    efk = jnp.exp(p[0]) * kernels.ExpSquared(jnp.exp(p[1]), distance=td) * kernels.Matern52(jnp.exp(p[2]), distance=efd)
    sfk = jnp.exp(p[3]) * kernels.ExpSquared(jnp.exp(p[4]), distance=td) * kernels.Matern52(jnp.exp(p[5]), distance=sfd)
    return efk + sfk
