from george.kernels import ExpSquaredKernel, ExpSine2Kernel, RationalQuadraticKernel, ConstantKernel
import numpy as np

#kernel = 0.378**2 * ExpSquaredKernel(48.5) * (ExpSine2Kernel(gamma=3.0, log_period=0.0) + NegativeConstantKernel(np.log(0.38))) + 0.0831**2 * ExpSquaredKernel(32.4) * (ExpSine2Kernel(gamma=3.03, log_period=np.log(0.5)) + ConstantKernel(np.log(0.305))) + 0.15**2 * ExpSquaredKernel(0.5) + 0.10**2 * RationalQuadraticKernel(metric=0.025, log_alpha=-1)

# kernel = 0.238**2 * ExpSquaredKernel(50) * (ExpSine2Kernel(gamma=10.355, log_period=0.0) + NegativeConstantKernel(np.log(0.146))) + 0.0388**2 * ExpSquaredKernel(40) * (ExpSine2Kernel(gamma=20.8, log_period=np.log(0.5)) + ConstantKernel(np.log(0.216))) + 0.0383**2 * ExpSquaredKernel(1.925) + 0.108**2 * RationalQuadraticKernel(metric=0.00164, log_alpha=-1.578)

# kernel = 0.195**2 * ExpSquaredKernel(2000) * ExpSine2Kernel(gamma=14.964, log_period=0.0) + 0.0232**2 * ExpSquaredKernel(127.597) * (ExpSine2Kernel(gamma=20.909, log_period=np.log(0.5)) + ConstantKernel(np.log(0.653))) + 0.0321**2 * ExpSquaredKernel(5.028) + 0.0848**2 * RationalQuadraticKernel(metric=0.00207, log_alpha=-0.11357)

#kernel = 0.191**2 * ExpSquaredKernel(1900) * ExpSine2Kernel(gamma=14.951, log_period=0.0) + 0.02227**2 * ExpSquaredKernel(133.358) * (ExpSine2Kernel(gamma=20.895, log_period=np.log(0.5)) + ConstantKernel(np.log(0.651))) + 0.0314**2 * ExpSquaredKernel(5.097) + 0.146**2 * RationalQuadraticKernel(metric=0.00207, log_alpha=-0.11357)

kernel = 0.197**2 * ExpSquaredKernel(2128) * ExpSine2Kernel(gamma=13.332, log_period=0.0) + 0.0247**2 * ExpSquaredKernel(133.163) * (ExpSine2Kernel(gamma=18.622, log_period=np.log(0.5)) + ConstantKernel(np.log(0.604))) + 0.0340**2 * ExpSquaredKernel(4.166) + 0.0771**2 * RationalQuadraticKernel(metric=0.00347, log_alpha=0.3611)

kernel.freeze_parameter('k1:k1:k1:k2:log_period')
kernel.freeze_parameter('k1:k1:k2:k2:k1:log_period')

delta_kernel = 0.108**2 * ExpSquaredKernel(972) * ExpSine2Kernel(gamma=23.548, log_period=0.0) + 0.0421**2 * ExpSquaredKernel(1e6) * (ExpSine2Kernel(gamma=2.270, log_period=np.log(0.5)) + ConstantKernel(np.log(0.000877))) + 0.0417**2 * ExpSquaredKernel(3.233) + 0.0737**2 * RationalQuadraticKernel(metric=0.00436, log_alpha=17.903)

delta_kernel.freeze_parameter('k1:k1:k1:k2:log_period')
delta_kernel.freeze_parameter('k1:k1:k2:k2:k1:log_period')
