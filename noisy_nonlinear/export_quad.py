#!/usr/bin/env python3

import logging

import coloredlogs
import numpy as np
import pandas as pd

from noisy_nonlinear.output import output_directory
from noisy_nonlinear.consts import const_delta
from noisy_nonlinear.func import NoisyFunc
from noisy_nonlinear.params import Params
from noisy_nonlinear.log import logger
from noisy_nonlinear.noisy_solver import solve
from noisy_nonlinear.quad import Quad


names = [('crit', "Criticality"),
         ('obj', "Objective"),
         ('red_ratio', "ReductionRatio"),
         ('delta_lp', "LPRadius"),
         ('opt_dist', "OptDist"),
         ('crit', "Criticality"),
         ('noisy_crit', "NoisyCriticality")]


output_filename = output_directory("quad")


def export_results(res_norm, res_stab):

    for (name, desc) in names:
        arrays = [res_norm.stats[name],
                  res_stab.stats[name]]

        columns = ["Classical", "Stabilized"]

        df = pd.DataFrame(np.vstack(arrays).T,
                          columns=columns)

        df.to_csv(output_filename(f"{desc}.csv"),
                  index=True,
                  index_label="Iteration",
                  sep=';')


if __name__ == "__main__":
    coloredlogs.install(level=logging.DEBUG)
    logger.info("Exporting quad data")

    failing_seed = 2

    value_error = 1e-1
    deriv_error = 1e-5
    penalty = 1e-2

    params = Params()

    func = Quad()

    x_0 = np.zeros((func.dim(),))
    x_0[0] = 1000.

    noisy_func = NoisyFunc(func,
                           value_error,
                           deriv_error,
                           seed=failing_seed)

    lip_deriv = func.deriv_lipschitz(None, None)

    delta = const_delta(noisy_func,
                        penalty,
                        lip_deriv,
                        params)

    print("Criticality bound: {0}".format(delta))

    normal_params = params.replace(stabilization=0.)

    res_norm = solve(noisy_func,
                     penalty,
                     x_0,
                     params=normal_params)

    res_norm.stats['red_ratio'] = np.clip(res_norm.stats['red_ratio'], -5., 5.)

    # Reset internal state (pseudo-randomness) of noisy function
    noisy_func = NoisyFunc(func,
                           value_error,
                           deriv_error,
                           seed=failing_seed)

    res_stab = solve(noisy_func,
                     penalty,
                     x_0,
                     params=params)

    res_stab.stats['red_ratio'] = np.clip(res_stab.stats['red_ratio'], -5., 5.)

    export_results(res_norm, res_stab)

    all_normal_obj = []
    all_stab_obj = []

    num_samples = 100

    for seed in range(num_samples):

        noisy_func = NoisyFunc(func,
                               value_error,
                               deriv_error,
                               seed=seed)

        res_norm = solve(noisy_func,
                         penalty,
                         x_0,
                         params=normal_params.replace(collect_stats=False))

        all_normal_obj.append(res_norm.noisy_obj)

        noisy_func = NoisyFunc(func,
                               value_error,
                               deriv_error,
                               seed=seed)

        res_stab = solve(noisy_func,
                         penalty,
                         x_0,
                         params=params.replace(collect_stats=False))

        all_stab_obj.append(res_stab.noisy_obj)

    columns = ["Classical", "Stabilized"]

    arrays = [np.array(all_normal_obj),
              np.array(all_stab_obj)]

    df = pd.DataFrame(np.vstack(arrays).T,
                      columns=columns)

    df.to_csv(output_filename("Comparison.csv"),
              index=True,
              index_label="Seed",
              sep=';')
