from george.kernels import ExpSquaredKernel, ExpSine2Kernel, RationalQuadraticKernel, ConstantKernel, NegativeConstantKernel
import numpy as np

#kernel = 0.378**2 * ExpSquaredKernel(48.5) * (ExpSine2Kernel(gamma=3.0, log_period=0.0) + NegativeConstantKernel(np.log(0.38))) + 0.0831**2 * ExpSquaredKernel(32.4) * (ExpSine2Kernel(gamma=3.03, log_period=np.log(0.5)) + ConstantKernel(np.log(0.305))) + 0.15**2 * ExpSquaredKernel(0.5) + 0.10**2 * RationalQuadraticKernel(metric=0.025, log_alpha=-1)

kernel = 0.238**2 * ExpSquaredKernel(50) * (ExpSine2Kernel(gamma=10.355, log_period=0.0) + NegativeConstantKernel(np.log(0.146))) + 0.0388**2 * ExpSquaredKernel(40) * (ExpSine2Kernel(gamma=20.8, log_period=np.log(0.5)) + ConstantKernel(np.log(0.216))) + 0.0383**2 * ExpSquaredKernel(1.925) + 0.108**2 * RationalQuadraticKernel(metric=0.00164, log_alpha=-1.578)

kernel.freeze_parameter('k1:k1:k1:k2:k1:log_period')
kernel.freeze_parameter('k1:k1:k2:k2:k1:log_period')
