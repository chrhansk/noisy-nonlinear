#!/usr/bin/env python3

import logging

import matplotlib
import matplotlib.pyplot as plt

import coloredlogs
import numpy as np
import pandas as pd

from noisy_nonlinear.output import output_directory
from noisy_nonlinear.log import logger
from noisy_nonlinear.crit import criticality
from noisy_nonlinear.func import NoisyFunc
from noisy_nonlinear.params import Params
from noisy_nonlinear.noisy_solver import solve
from noisy_nonlinear.rosenbrock import Rosenbrock

output_filename = output_directory("rosenbrock")

vmin = -2.
vmax = 2.
num_points = 1001

penalty = 0.1

aspect_ratio = 3. / 4.

x0 = np.array([-1.5, 0.])
deriv_error = 1e-5
value_errors = [1e-2, 1e-1, 1e0, 1e1]


def graymap(num_levels, lower):
    colors = np.linspace(lower, 1., num_levels)

    colors = np.tile(colors, (3, 1))

    values = np.hstack([colors.T, np.ones((num_levels,))[:, None]])

    return matplotlib.colors.ListedColormap(values)


def trajectory_from_result(res):
    return np.vstack(res.stats['primal'])


def export_criticality_plot(x, y, z, all_traj, filename):

    start = 0
    stop = 12
    base = 2

    levels = np.logspace(start=start, stop=stop, base=base)
    num_levels = levels.size

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(aspect_ratio)

    cmap = graymap(num_levels, 0.3)

    plt.contourf(x, y, z + 1., levels=levels, cmap=cmap)

    log_norm = matplotlib.colors.LogNorm(vmin=base**start, vmax=base**stop)

    colorbar = fig.colorbar(plt.cm.ScalarMappable(log_norm, cmap=cmap),
                            ax=ax,
                            location='left')

    cax = colorbar.ax
    (l, b, w, h) = ax.get_position().bounds
    (lc, bc, wc, hc) = cax.get_position().bounds

    # Set yval / height of colorbar axes to match
    # that of the main axes (with given aspect ratio)
    cax.set_position([lc, b, wc, h])

    labels = []
    lines = []

    markers = ["s", ".", "D", "p"]
    linestyles = ["dotted", "dashed", "dashdot", "solid"]

    zindices = {"s": 3,
                ".": 10,
                "D": 3,
                "p": 5}

    for i, (value_error, res) in enumerate(all_traj):
        values = trajectory_from_result(res)
        x = values[:, 0]
        y = values[:, 1]

        line, = plt.plot(x, y, 'k', linewidth=1, linestyle=linestyles[i])
        xfinal = x[-1]
        yfinal = y[-1]

        marker = markers[i]
        zindex = zindices[marker]

        line, = plt.plot(xfinal, yfinal,
                         color="black",
                         linestyle=linestyles[i],
                         marker=markers[i],
                         markerfacecolor="white",
                         markeredgecolor="black",
                         ms=8,
                         zorder=zindex)

        lines.append(line)
        labels.append(str(value_error))

    ax.legend(lines, labels, loc='lower right')
    # plt.show()
    plt.savefig(output_filename(filename), bbox_inches="tight")


def export_criticality(x, y, z, filename):
    arrays = [x.ravel(), y.ravel(), z.ravel()]

    df = pd.DataFrame(np.vstack(arrays).T,
                      columns=["x", "y", "z"])

    output = output_filename(filename)

    df.to_csv(output,
              index=False,
              header=False,
              sep=' ')


def import_criticality(filename):
    logger.info(f"Importing criticality from '{filename}'")
    df = pd.read_csv(filename,
                     # index=False,
                     header=None,
                     sep=' ')

    arr = df.to_numpy()

    x = arr[:, 0]
    y = arr[:, 1]
    z = arr[:, 2]

    xx = x.reshape((num_points, num_points))
    yy = y.reshape((num_points, num_points))
    zz = z.reshape((num_points, num_points))

    return (xx, yy, zz)


def export_trajectory(res, params, value_error):
    values = trajectory_from_result(res)

    x = values[:, 0]
    y = values[:, 1]

    df = pd.DataFrame(np.vstack([x, y]).T,
                      columns=["x", "y"])

    if params.stabilization is None:
        output = output_filename(f"Trajectory_nostab_{value_error}.dat")
    else:
        output = output_filename(f"Trajectory_stab_{value_error}.dat")

    df.to_csv(output,
              index=False,
              header=False,
              sep=' ')


def export_trajectories(func, params, xx, yy, zz):
    all_trajectories = []

    for value_error in value_errors:

        logger.info(f"Solving with error {value_error}")

        noisy_func = NoisyFunc(func,
                               value_error,
                               deriv_error)

        res = solve(noisy_func,
                    penalty,
                    x0,
                    params=params)

        export_trajectory(res, params, value_error)

        all_trajectories.append((value_error, res))

    filename = None

    if params.is_stabilized():
        filename = "Stabilized.pdf"
    else:
        filename = "Classical.pdf"

    export_criticality_plot(xx,
                            yy,
                            zz,
                            all_trajectories,
                            filename)


def collect_criticality(func, x, y):
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)

    logger.info("Collecting criticality")

    for i in range(len(x)):
        for j in range(len(y)):
            p = np.array([x[i], y[j]])

            func_val = func.value(p)
            deriv_val = func.deriv_sparse(p)

            zz[i, j] = criticality(func,
                                   func_val,
                                   deriv_val,
                                   penalty)

    return xx, yy, zz


def main():
    coloredlogs.install(level=logging.DEBUG)
    logger.info("Exporting rosenbrock data")

    matplotlib.rcParams['mathtext.fontset'] = 'cm'

    func = Rosenbrock()
    params = Params()

    x = np.linspace(vmin, vmax, num_points)
    y = np.linspace(vmin, vmax, num_points)

    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)

    # (xx, yy, zz) = import_criticality("output/rosenbrock/Criticality.dat")

    (xx, yy, zz) = collect_criticality(func, x, y)
    export_criticality(xx, yy, zz, "Criticality.dat")

    export_trajectories(func,
                        params.replace(stabilization=0.),
                        xx,
                        yy,
                        zz)

    export_trajectories(func,
                        params.replace(stabilization=None),
                        xx,
                        yy,
                        zz)


if __name__ == "__main__":
    main()
