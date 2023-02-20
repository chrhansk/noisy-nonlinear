import matplotlib.pyplot as plt
import numpy as np

from crit import criticality


def plot_results(res, func, penalty, title=None):
    x_opt = func.opt()
    red_ratio = res.stats['red_ratio']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))
    ax1.plot(red_ratio)
    ax1.set_title("Reduction ratio")

    primals = res.stats['primal']

    ax2.semilogy(np.linalg.norm(np.stack(primals) - x_opt, axis=1))
    ax2.set_title("Distance to optimum")

    delta = res.stats['delta_lp']

    ax3.semilogy(delta)
    ax3.set_title("LP trust region radius")

    crits = []

    for primal in primals:
        func_val = func.value(primal)
        deriv_val = func.deriv(primal)
        crits.append(criticality(func, func_val, deriv_val, penalty))

    ax4.semilogy(crits)
    ax4.set_title("Noiseless criticality")

    fig.suptitle(title, fontsize='xx-large')


def plot_compare(res_norm, res_stab, func, penalty, crit_bound=None):
    x_opt = func.opt()

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(20, 20))

    red_ratio = np.array(res_norm.stats['red_ratio'])
    ax1.plot(np.clip(red_ratio, -5., 5.), marker='o')

    red_ratio = np.array(res_stab.stats['red_ratio'])
    ax1.plot(np.clip(red_ratio, -5., 5.), marker='o')
    ax1.set_title("(Clipped) reduction ratio")

    primals = res_norm.stats['primal']
    ax2.semilogy(np.linalg.norm(np.stack(primals) - x_opt, axis=1))

    primals = res_stab.stats['primal']
    ax2.semilogy(np.linalg.norm(np.stack(primals) - x_opt, axis=1))

    ax2.set_title("Distance to optimum")

    delta = res_norm.stats['delta_lp']
    ax3.semilogy(delta)

    delta = res_stab.stats['delta_lp']
    ax3.semilogy(delta)

    ax3.set_title("LP trust region radius")

    ax4.semilogy(res_norm.stats['crit'])

    ax4.semilogy(res_stab.stats['crit'])

    ax4.set_title("Noiseless criticality")

    ax5.semilogy(res_norm.stats['noisy_crit'])

    ax5.semilogy(res_stab.stats['noisy_crit'])

    if crit_bound is not None:
        ax5.axhline(y=crit_bound, color='black')

    ax5.set_title("Noisy criticality")

    ax6.plot(res_norm.stats['obj'])

    ax6.plot(res_stab.stats['obj'])

    ax6.set_title("Objective")

    lines, _ = fig.axes[-1].get_legend_handles_labels()

    fig.legend(lines, labels=["Classical", "Stabilized"])

    fig.suptitle("Comparison", fontsize='xx-large')
