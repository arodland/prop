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
    'fof2': [0.24980929837457075, 0.95, 91.31127252290733, 54.993966946728015, -4.0193854412596295, -1.5259456951692278, -1.7163515088523236, -3.3322476315640213, 14.2056571165412, -0.7489449656306164, -3.453619259116425, 4.644690567941146, -0.42844142169629373],
    'hmf2': [0.2665792994289858, 0.95, 92.58784404907338, 62.39133577755096, -3.9088597137507683, -1.4858225478390632, -1.6947451541959688, -1.7001159856137529, 12.911574146570649, -0.2986819719173836, 1.1576681467980094, 8.269483246664013, 1.086820641550241],
    'mufd': [0.2555750674231012, 0.95, 92.00261150345762, 58.350738272081536, -3.9372415118060107, -1.507979175787714, -1.735751877879807, -2.906213821539421, 15.596492883285109, -0.5561632379764008, -1.064865783839084, 6.565029099213679, 0.5531616561618322],
    'md': [0.24764545444423944, 0.95, 92.56918822229439, 56.947357857198135, -3.841427265825709, -1.3864987927237369, -1.680535743422548, -1.7150693161516657, 14.206442670686025, -0.20329595337163225, -0.9212713363615613, 6.617514655405649, 0.395040183517625],
}
