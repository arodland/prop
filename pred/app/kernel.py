from george.kernels import ExpSquaredKernel, ExpSine2Kernel, RationalQuadraticKernel, ConstantKernel
import numpy as np

#kernel = 0.378**2 * ExpSquaredKernel(48.5) * (ExpSine2Kernel(gamma=3.0, log_period=0.0) + NegativeConstantKernel(np.log(0.38))) + 0.0831**2 * ExpSquaredKernel(32.4) * (ExpSine2Kernel(gamma=3.03, log_period=np.log(0.5)) + ConstantKernel(np.log(0.305))) + 0.15**2 * ExpSquaredKernel(0.5) + 0.10**2 * RationalQuadraticKernel(metric=0.025, log_alpha=-1)

# kernel = 0.238**2 * ExpSquaredKernel(50) * (ExpSine2Kernel(gamma=10.355, log_period=0.0) + NegativeConstantKernel(np.log(0.146))) + 0.0388**2 * ExpSquaredKernel(40) * (ExpSine2Kernel(gamma=20.8, log_period=np.log(0.5)) + ConstantKernel(np.log(0.216))) + 0.0383**2 * ExpSquaredKernel(1.925) + 0.108**2 * RationalQuadraticKernel(metric=0.00164, log_alpha=-1.578)

# kernel = 0.195**2 * ExpSquaredKernel(2000) * ExpSine2Kernel(gamma=14.964, log_period=0.0) + 0.0232**2 * ExpSquaredKernel(127.597) * (ExpSine2Kernel(gamma=20.909, log_period=np.log(0.5)) + ConstantKernel(np.log(0.653))) + 0.0321**2 * ExpSquaredKernel(5.028) + 0.0848**2 * RationalQuadraticKernel(metric=0.00207, log_alpha=-0.11357)

#kernel = 0.191**2 * ExpSquaredKernel(1900) * ExpSine2Kernel(gamma=14.951, log_period=0.0) + 0.02227**2 * ExpSquaredKernel(133.358) * (ExpSine2Kernel(gamma=20.895, log_period=np.log(0.5)) + ConstantKernel(np.log(0.651))) + 0.0314**2 * ExpSquaredKernel(5.097) + 0.146**2 * RationalQuadraticKernel(metric=0.00207, log_alpha=-0.11357)

kernel = 0.197**2 * ExpSquaredKernel(2128) * ExpSine2Kernel(gamma=13.332, log_period=0.0) + 0.0247**2 * ExpSquaredKernel(133.163) * (ExpSine2Kernel(gamma=18.622, log_period=np.log(0.5)) + ConstantKernel(np.log(0.604))) + 0.0340**2 * ExpSquaredKernel(4.166) + 0.0771**2 * RationalQuadraticKernel(metric=0.00347, log_alpha=0.3611)

kernel.freeze_parameter('k1:k1:k1:k2:log_period')
kernel.freeze_parameter('k1:k1:k2:k2:k1:log_period')

# delta_kernel = 0.108**2 * ExpSquaredKernel(972) * ExpSine2Kernel(gamma=23.548, log_period=0.0) + 0.0421**2 * ExpSquaredKernel(1e6) * (ExpSine2Kernel(gamma=2.270, log_period=np.log(0.5)) + ConstantKernel(np.log(0.000877))) + 0.0417**2 * ExpSquaredKernel(3.233) + 0.0737**2 * RationalQuadraticKernel(metric=0.00436, log_alpha=17.903)

# delta_kernel = 0.109**2 * ExpSquaredKernel(689) * ExpSine2Kernel(gamma=22.638, log_period=0.0) + 0.00554**2 * ExpSquaredKernel(1e6) * (ExpSine2Kernel(gamma=5.667, log_period=np.log(0.5)) + ConstantKernel(np.log(0.000845))) + 0.0483**2 * ExpSquaredKernel(2.788) + 0.0757**2 * RationalQuadraticKernel(metric=0.00424, log_alpha=17.903) # 2021-04-04

# delta_kernel = 0.109**2 * ExpSquaredKernel(689) * ExpSine2Kernel(gamma=22.638, log_period=0.0) + 0.00554**2 * ExpSquaredKernel(1e6) * (ExpSine2Kernel(gamma=5.667, log_period=np.log(0.5)) + ConstantKernel(np.log(0.000845))) + 0.0483**2 * ExpSquaredKernel(2.788) + 0.0757**2 * RationalQuadraticKernel(metric=0.00424, log_alpha=17.903) # 2021-04-05

# delta_kernel = 0.109**2 * ExpSquaredKernel(689) * ExpSine2Kernel(gamma=22.638, log_period=0.0) + 0.0483**2 * ExpSquaredKernel(2.788) + 0.0757**2 * RationalQuadraticKernel(metric=0.00424, log_alpha=17.903) # 2021-04-05.2

# delta_kernel = 0.0917**2 * ExpSquaredKernel(611) * ExpSine2Kernel(gamma=30.976, log_period=0.0) + 0.0813**2 * ExpSquaredKernel(10.105) + 0.100**2 * RationalQuadraticKernel(metric=0.00175, log_alpha=-1.106) + ConstantKernel(np.log(0.04)) # 2021-04-05.3

delta_kernel = 0.0971**2 * ExpSquaredKernel(611) * ExpSine2Kernel(gamma=29.696, log_period=0.0) + 0.0444**2 * ExpSquaredKernel(8.135) + 0.0928**2 * RationalQuadraticKernel(metric=0.00182, log_alpha=-0.700) + ConstantKernel(np.log(0.00162)) # 2021-04-06

for param in delta_kernel.get_parameter_names():
    if param.endswith(':log_period'):
        delta_kernel.freeze_parameter(param)
#delta_kernel = 0.0971**2 * ExpSquaredKernel(8) * ExpSine2Kernel(gamma=29.696, log_period=0.0) + 0.0444**2 * ExpSquaredKernel(4) + 0.0928**2 * RationalQuadraticKernel(metric=0.00182, log_alpha=-0.700) + ConstantKernel(np.log(0.00162)) # 2022-06-29

delta_fof2_kernel = 0.0830**2 * ExpSquaredKernel(1141) * ExpSine2Kernel(gamma=35.368, log_period=0.0) + 0.04388**2 * ExpSquaredKernel(19.116) + 0.1105**2 * RationalQuadraticKernel(metric=0.00376, log_alpha=-1.224) + ConstantKernel(np.log(0.000548))

for param in delta_fof2_kernel.get_parameter_names():
    if param.endswith(':log_period'):
        delta_fof2_kernel.freeze_parameter(param)

delta_mufd_kernel = 0.0879**2 * ExpSquaredKernel(1040) * ExpSine2Kernel(gamma=38.792, log_period=0.0) + 0.03511**2 * ExpSquaredKernel(6.2735) + 0.1212**2 * RationalQuadraticKernel(metric=0.00171, log_alpha=-1.260) + ConstantKernel(np.log(0.000348))

for param in delta_mufd_kernel.get_parameter_names():
    if param.endswith(':log_period'):
        delta_mufd_kernel.freeze_parameter(param)

delta_hmf2_kernel = 0.0407**2 * ExpSquaredKernel(1126) * ExpSine2Kernel(gamma=20.511, log_period=0.0) + 0.02785**2 * ExpSquaredKernel(44.78) + 0.0622**2 * RationalQuadraticKernel(metric=0.00045, log_alpha=-0.759) + ConstantKernel(np.log(0.000204))

for param in delta_mufd_kernel.get_parameter_names():
    if param.endswith(':log_period'):
        delta_mufd_kernel.freeze_parameter(param)

