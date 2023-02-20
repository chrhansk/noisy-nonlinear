#!/usr/bin/env python3

import logging
import coloredlogs

import numpy as np
import pandas as pd

from noisy_nonlinear.log import logger
from noisy_nonlinear.func import omega
from noisy_nonlinear.output import output_directory
from noisy_nonlinear.consts import stabilization
from noisy_nonlinear.func import NoisyFunc, feas_res
from noisy_nonlinear.HS71 import HS71
from noisy_nonlinear.params import Params
from noisy_nonlinear.noisy_solver import solve

output_filename = output_directory("HS71_stats")

deriv_error = 0.

noise_levels = [0., 1e-2, 1e-1]
penalties = [1e0, 1e1, 1e2, 1e3]

base = 2
min_power = -7


def solve_stab(noise_level, penalty, prescribed_stabilization):

    orig_func = HS71()

    func = NoisyFunc(orig_func, noise_level, deriv_error)

    params = Params(perform_eqp=False,
                    max_it=100,
                    collect_stats=True,
                    stabilization=prescribed_stabilization)

    x_0 = np.copy(orig_func.x_0)

    res = solve(func,
                penalty,
                x_0,
                params=params)

    return res


def solve_and_export(noise_level, penalty):

    logger.info(f"Evaluating stabilization values for noise level {noise_level}, penalty {penalty}")

    all_stabilizations = []
    all_opt_dists = []
    all_noisy_obj_values = []
    all_orig_obj_values = []

    all_feas_res = []
    all_noisy_feas_res = []
    all_crit = []
    all_noisy_crit = []

    orig_func = HS71()

    def add_values(stab,
                   res):
        all_stabilizations.append(stab)
        all_opt_dists.append(np.linalg.norm(res.x - orig_func.x_opt))
        all_noisy_obj_values.append(res.noisy_obj)
        all_orig_obj_values.append(omega(orig_func, orig_func.value(res.x), penalty))

        all_feas_res.append(feas_res(orig_func, orig_func.value(res.x)))

        all_noisy_feas_res.append(feas_res(orig_func, res.func_val))

        all_crit.append(res.stats['crit'][-1])
        all_noisy_crit.append(res.stats['noisy_crit'][-1])

    stab = 0

    res_norm = solve_stab(noise_level, penalty, stab)

    add_values(stab, res_norm)

    power = min_power

    while True:
        stab = base**power
        logger.info(f"Solving problem with stabilization {stab}")
        res = solve_stab(noise_level,
                         penalty,
                         stab)
        add_values(stab, res)

        power += 1

        if res.num_rejected == 0:
            break

    arrays = [all_stabilizations,
              all_opt_dists,
              all_noisy_obj_values,
              all_orig_obj_values,
              all_feas_res,
              all_noisy_feas_res,
              all_crit,
              all_noisy_crit]

    columns = ["Stabilization",
               "OptDist",
               "NoisyObjVal",
               "ObjVal",
               "FeasRes",
               "NoisyFeasRes",
               "Crit",
               "NoisyCrit"]

    df = pd.DataFrame(np.vstack(arrays).T,
                      columns=columns)

    filename = f"Results_{noise_level}_{penalty}.csv"

    df.to_csv(output_filename(filename),
              index=False,
              sep=';')


if __name__ == "__main__":
    coloredlogs.install(level=logging.DEBUG)
    logger.info("Exporting HS71_stats data")

    for noise_level in noise_levels:
        for penalty in penalties:
            params = Params()
            orig_func = HS71()
            func = NoisyFunc(orig_func, noise_level, deriv_error)
            stab = stabilization(func, penalty, params)

            logger.info(f"Required stabilization for noise level {noise_level}, penalty {penalty}: {stab}")

    for noise_level in noise_levels:
        for penalty in penalties:
            solve_and_export(noise_level, penalty)
