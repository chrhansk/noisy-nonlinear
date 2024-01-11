#!/usr/bin/env python3

import numpy as np
import pandas as pd

from noisy_nonlinear.output import output_directory
from noisy_nonlinear.func import omega
from noisy_nonlinear.log import logger
from noisy_nonlinear.tvdenoiser import TVDenoiser
from noisy_nonlinear.noisy_solver import solve
from noisy_nonlinear.params import Params

output_filename = output_directory("tvdenoiser")

noise_levels = [0.01, 0.05, 0.1]

penalty = 0.005

base = 2
min_power = -7

image_filename = "image_512.png"


def solve_and_export_orig():
    logger.info("Solving noiseless problem")

    solve_stab(noise_level=0.,
               prescribed_stabilization=0.,
               output_template="noiseless")


def solve_stab(noise_level, prescribed_stabilization, output_template=None):

    params = Params(perform_eqp=False,
                    max_it=100,
                    collect_stats=False,
                    stabilization=prescribed_stabilization)

    func = TVDenoiser(image_filename, noise_level)

    x_0 = np.zeros((func.dim(),))

    res = solve(func,
                penalty,
                x_0,
                params=params)

    if output_template is None:
        output_template = name_stub(noise_level, prescribed_stabilization)

    output = f"Result_{output_template}.png"

    func.as_image(res.x).save(output_filename(output))

    return res


def name_stub(noise_level, stabilization):
    return f"{noise_level}_{stabilization}"


def solve_and_export(noise_level):

    all_stabilizations = []
    all_noisy_obj_values = []
    all_orig_obj_values = []
    all_fidelity_values = []

    logger.info(f"Evaluating stabilization values for noise level {noise_level}")

    orig_func = TVDenoiser(image_filename, noise_level=0.)

    def add_values(stabilization,
                   res):
        all_stabilizations.append(stabilization)
        all_noisy_obj_values.append(res.noisy_obj)
        all_orig_obj_values.append(omega(orig_func, orig_func.value(res.x), penalty))
        all_fidelity_values.append(orig_func.fidelity(res.x))

        print(omega(orig_func, orig_func.value(res.x), penalty),
              orig_func.fidelity(res.x),
              omega(orig_func, orig_func.value(res.x), penalty) - orig_func.fidelity(res.x))

    stabilization = 0.

    res_norm = solve_stab(noise_level,
                          stabilization)

    add_values(stabilization, res_norm)

    power = min_power

    while True:
        stabilization = base**power
        logger.info(f"Solving problem with stabilization {stabilization}")
        res = solve_stab(noise_level,
                         stabilization)
        add_values(stabilization, res)

        power += 1

        if res.num_rejected == 0:
            break

    arrays = [all_stabilizations,
              all_noisy_obj_values,
              all_orig_obj_values,
              all_fidelity_values]

    columns = ["Stabilization",
               "NoisyObjVal",
               "ObjVal",
               "Fidelity"]

    df = pd.DataFrame(np.vstack(arrays).T,
                      columns=columns)

    filename = f"Results_{noise_level}.csv"

    df.to_csv(output_filename(filename),
              index=False,
              sep=';')


def print_required_stabilizations():
    from noisy_nonlinear.consts import stabilization

    params = Params(perform_eqp=False,
                    max_it=100,
                    collect_stats=False,
                    stabilization=None)

    for noise_level in noise_levels:
        func = TVDenoiser(image_filename, noise_level)

        required_stabilization = stabilization(func, penalty, params)

        logger.info(f"Required stabilization for noise level {noise_level}: {required_stabilization}")


if __name__ == "__main__":
    from multiprocessing import Pool, cpu_count
    import coloredlogs
    coloredlogs.install()
    logger.info("Exporting tvdenoiser data")

    print_required_stabilizations()

    solve_and_export_orig()

    with Pool(cpu_count()) as pool:
        pool.map(solve_and_export, noise_levels)
