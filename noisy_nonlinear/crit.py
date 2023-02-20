import numpy as np
import matplotlib.pyplot as plt

from noisy_nonlinear.consts import const_delta
from noisy_nonlinear.lp import solve_lp
from noisy_nonlinear.params import Params
from noisy_nonlinear.func import NoisyFunc, linearized_red
from noisy_nonlinear.rosenbrock import Rosenbrock


def criticality(func, func_val, deriv_val, penalty):
    (step, fun) = solve_lp(func,
                           func_val,
                           deriv_val,
                           penalty,
                           1.)

    return linearized_red(func,
                          func_val,
                          deriv_val,
                          penalty,
                          step)


if __name__ == "__main__":
    penalty = 0.1

    vmin = -1.
    vmax = 1.

    params = Params()
    func = Rosenbrock()

    num_points = 100

    lip_deriv = func.deriv_lipschitz(np.array([vmin, vmin]),
                                     np.array([vmax, vmax]))

    value_error = .1
    deriv_error = .1

    noisy_func = NoisyFunc(func,
                           value_error,
                           deriv_error)

    delta = const_delta(noisy_func,
                        penalty,
                        lip_deriv,
                        params)

    print("Criticality bound: {0}".format(delta))

    x = np.linspace(vmin, vmax, num_points + 1)
    y = np.linspace(vmin, vmax, num_points + 1)

    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)

    for i in range(len(x)):
        for j in range(len(y)):
            p = np.array([x[i], y[j]])

            func_val = func.value(p)
            deriv_val = func.deriv(p)

            zz[i, j] = criticality(func_val,
                                   deriv_val,
                                   penalty)

    h = plt.contour(x, y, np.log(zz + 1.))

    plt.axis('scaled')

    plt.colorbar()

    plt.show()
