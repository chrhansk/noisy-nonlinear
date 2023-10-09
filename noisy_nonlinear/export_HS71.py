#!/usr/bin/env python3

import logging
import coloredlogs

import numpy as np
import pandas as pd

from noisy_nonlinear.output import output_directory
from noisy_nonlinear.consts import stabilization
from noisy_nonlinear.func import NoisyFunc, feas_res
from noisy_nonlinear.HS71 import HS71
from noisy_nonlinear.log import logger
from noisy_nonlinear.params import Params
from noisy_nonlinear.noisy_solver import solve

output_filename = output_directory("HS71")

deriv_error = 0.

noise_levels = [0., 1e-2, 1e-1]
penalties = [1e0, 1e1, 1e2, 1e3]


def solve_and_export(stabilized):
    func = HS71()

    stab = None if stabilized else 0.

    stab_name = "stabilized" if stabilized else "classical"

    params = Params(perform_eqp=False,
                    max_it=100,
                    stabilization=stab)

    x_0 = np.copy(func.x_0)

    columns = ["ValueError",
               "Penalty",
               "ReqStab",
               "OptDist",
               "FeasRes",
               "NoisyFeasRes",
               "Crit",
               "NoisyCrit",
               "Iterations",
               "NumAccept",
               "NumReject",
               "Termination"]

    df_penalties = [pd.DataFrame(columns=columns) for _ in penalties]

    df_all = pd.DataFrame(columns=columns)

    for noise_level in noise_levels:

        df = pd.DataFrame(columns=columns)

        for (i, penalty) in enumerate(penalties):

            noisy_func = NoisyFunc(func, noise_level, deriv_error)

            res = solve(noisy_func,
                        penalty,
                        x_0,
                        params=params)

            x = res.x

            required_stab = stabilization(noisy_func, penalty, params)

            opt_dist = np.linalg.norm(res.x - func.x_opt)

            orig_feas_res = feas_res(func,
                                     func.value(x))

            noisy_feas_res = feas_res(noisy_func,
                                      noisy_func.value(x))

            orig_crit = res.stats['crit'][-1]
            noisy_crit = res.stats['noisy_crit'][-1]

            values = [noise_level,
                      penalty,
                      required_stab,
                      opt_dist,
                      orig_feas_res,
                      noisy_feas_res,
                      orig_crit,
                      noisy_crit,
                      res.iterations,
                      res.num_accepted,
                      res.num_rejected,
                      res.termination]

            df.loc[len(df)] = values

            df_penalty = df_penalties[i]

            df_penalty.loc[len(df_penalty)] = values

            df_all.loc[len(df_all)] = values

        df.to_csv(output_filename(f"Stats_{stab_name}_noise_level_{noise_level}.csv"),
                  sep=';',
                  index=False)

    for (i, penalty) in enumerate(penalties):
        df_penalty = df_penalties[i]

        df_penalty.to_csv(output_filename(f"Stats_{stab_name}_penalty_{penalty}.csv"),
                          sep=';',
                          index=False)

    df_all.to_csv(output_filename(f"Stats_{stab_name}.csv"),
                  sep=';',
                  index=False)


if __name__ == "__main__":
    coloredlogs.install(level=logging.DEBUG)
    logger.info("Exporting HS71 data")

    solve_and_export(stabilized=True)
    solve_and_export(stabilized=False)
