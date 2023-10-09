#!/usr/bin/env python3

import logging

import coloredlogs
import numpy as np
import pandas as pd

from noisy_nonlinear.output import output_directory
from noisy_nonlinear.log import logger
from noisy_nonlinear.func import NoisyFunc
from noisy_nonlinear.params import Params
from noisy_nonlinear.noisy_solver import solve
from noisy_nonlinear.rosenbrock import Rosenbrock

output_filename = output_directory("rosenbrock_stats")

vmin = -2.
vmax = 2.
num_points = 11

penalty = 0.1

x0 = np.array([-1.5, 0.])
deriv_error = 1e-5
value_errors = [1e-2, 1e-1, 1e0, 1e1]

num_seeds = 100


def export_dist(func, all_res_norm, all_res_stab, value_error):
    all_primal_norm = np.array([res.x for res in all_res_norm])
    all_primal_stab = np.array([res.x for res in all_res_stab])

    all_opt_dist_norm = np.linalg.norm(all_primal_norm - func.x_opt,
                                       axis=1)

    all_opt_dist_stab = np.linalg.norm(all_primal_stab - func.x_opt,
                                       axis=1)

    columns = ["Classical", "Stabilized"]

    df = pd.DataFrame(np.vstack([all_opt_dist_norm, all_opt_dist_stab]).T,
                      columns=columns)

    filename = f"Opt_distance_{value_error}.dat"

    df.to_csv(output_filename(filename),
              index=True,
              index_label="Seed",
              sep=';')


if __name__ == "__main__":
    coloredlogs.install(level=logging.DEBUG)
    logger.info("Exporting rosenbrock_stats data")

    func = Rosenbrock()
    params = Params(collect_stats=False)

    for value_error in value_errors:

        logger.info(f"Solving with error {value_error}")

        all_res_stab = []
        all_res_norm = []

        for seed in range(num_seeds):
            noisy_func = NoisyFunc(func,
                                   value_error,
                                   deriv_error,
                                   seed=seed)

            res_norm = solve(noisy_func,
                             penalty,
                             x0,
                             params=params.replace(stabilization=0.))

            all_res_norm.append(res_norm)

            noisy_func = NoisyFunc(func,
                                   value_error,
                                   deriv_error,
                                   seed=seed)

            res_stab = solve(noisy_func,
                             penalty,
                             x0,
                             params=params.replace(stabilization=None))

            all_res_stab.append(res_stab)

        export_dist(func, all_res_norm, all_res_stab, value_error)
