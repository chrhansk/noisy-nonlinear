#!/usr/bin/env python

import collections

import numpy as np

from noisy_nonlinear.bfgs import NoisyBFGS
from noisy_nonlinear.consts import stabilization
from noisy_nonlinear.crit import criticality
from noisy_nonlinear.eqp import solve_eqp
from noisy_nonlinear.func import omega, linearized_obj, linearized_red
from noisy_nonlinear.log import logger
from noisy_nonlinear.lp import solve_lp
from noisy_nonlinear.params import Params


Result = collections.namedtuple('Result', ['x',
                                           'func_val',
                                           'deriv_val',
                                           'termination',
                                           'noisy_obj',
                                           'iterations',
                                           'num_accepted',
                                           'num_rejected',
                                           'stats'])


def norm_lp(step):
    return np.linalg.norm(step, np.inf)


class Stats:
    def __init__(self):
        self._values = collections.defaultdict(lambda: [])

    def append(self, k, v):
        self._values[k].append(v)

    def append_func(self, k, f):
        self._values[k].append(f())

    def __getitem__(self, k):
        return self._values[k]

    def __setitem__(self, k, v):
        self._values[k] = v

    def update(self, k, f):
        self._values[k] = f(self._values[k])

    @property
    def values(self):
        return self._values


class NoStats:
    def append(self, k, v):
        pass

    def append_func(self, k, f):
        pass

    def __getitem__(self, k):
        return None

    def __setitem__(self, k, v):
        pass

    def update(self, k, f):
        pass

    @property
    def values(self):
        return dict()


def check_deriv(func, params, x, deriv_val):
    n = x.size

    deriv_val = deriv_val.toarray()

    deriv_pert = params.deriv_pert

    func_val = func.value(x)

    for j in range(n):

        next_x = np.copy(x)
        next_x[j] += deriv_pert

        next_func_val = func.value(next_x)

        deriv_col = 1./deriv_pert * (next_func_val - func_val)

        assert np.allclose(deriv_col,
                           deriv_val[:, j],
                           rtol=params.deriv_tol)


def solve(func, penalty, x_0, params=Params()):

    if params.collect_stats:
        stats = Stats()
    else:
        stats = NoStats()

    delta_lp = params.delta_lp_0
    delta = params.delta_0
    x = x_0

    if params.use_quasi_newton:
        bfgs = NoisyBFGS(func, params)
    else:
        bfgs = None

    func_val = func.value(x)
    deriv_val = func.deriv_sparse(x).tocsc()

    if params.stabilization is None:
        stab = stabilization(func, penalty, params)
        logger.info(f"Solving with required stabilization of {stab}")
    else:
        stab = params.stabilization

        if stab == 0.:
            logger.info("Solving without stabilization")
        else:
            logger.info(f"Solving with prescribed stabilization of {stab}")

    if stab == 0.:
        logger.info("Starting optimization without stabilization")
    else:
        logger.info(f"Starting optimization, stabilization: {stab}")

    logger.info(f"{'Iteration':>10}|{'Objective':>20}|{'Ratio':>20}|{'Type':>10}")

    x_opt = func.opt()

    orig_func = func.orig_func()

    num_accepted = 0
    num_rejected = 0

    curr_obj = None

    termination = "iteration limit"

    num_iterations = 0

    for it in range(params.max_it + 1):

        if params.deriv_check and not(func.is_noisy()):
            check_deriv(func, params, x, deriv_val)

        stats.append_func('primal',
                          lambda: np.copy(x))

        stats.append_func('opt_dist',
                          lambda: np.linalg.norm(x - x_opt))

        stats.append_func('noisy_crit',
                          lambda: criticality(func, func_val, deriv_val, penalty))

        stats.append_func('crit',
                          lambda: criticality(orig_func,
                                              orig_func.value(x),
                                              orig_func.deriv_sparse(x),
                                              penalty))

        curr_obj = omega(func, func_val, penalty)

        stats.append_func('obj',
                          lambda: omega(func, orig_func.value(x), penalty))

        if bfgs:
            hess_matrix = bfgs.get_matrix()
        else:
            hess_matrix = func.hess_sparse(x)

        def quadratic_term(step):
            return .5 * np.dot(step, hess_matrix.dot(step))

        def quadratic_obj(step):
            lin_obj = linearized_obj(func, func_val, deriv_val, penalty, step)

            return lin_obj + quadratic_term(step)

        # Compute LP step
        (lp_step, lp_obj) = solve_lp(func,
                                     func_val,
                                     deriv_val,
                                     penalty,
                                     delta_lp)

        alpha = min(1., delta / np.linalg.norm(lp_step))
        cauchy_step = alpha * lp_step

        crit_bound = linearized_red(func, func_val, deriv_val, penalty, lp_step)
        crit_bound *= max(1., 1./delta_lp)

        if delta_lp <= 1e-10:
            logger.warning("Trust region collapse")
            termination = "trust region collapse"
            break
        if abs(crit_bound) <= 1e-6:
            logger.info("Achieved optimality ({0})".format(crit_bound))
            logger.info(f"{it:>10}|{curr_obj:20g}")
            termination = "achieved optimality"
            break

        #assert crit_bound >= 0.

        for i in range(30):
            quad_term = quadratic_term(cauchy_step)
            lin_red = linearized_red(func, func_val, deriv_val, penalty, cauchy_step)

            if (1 - params.cauchy_eta) * lin_red >= quad_term:
                break
            else:
                alpha *= params.cauchy_tau
                cauchy_step = alpha * lp_step
        else:
            raise Exception("Line search failed to converge")

        logger.debug(f"Line search converged after {i} iterations, alpha = {alpha}")

        if params.perform_eqp:
            eqp_step = solve_eqp(func,
                                 cauchy_step,
                                 func_val,
                                 deriv_val,
                                 hess_matrix,
                                 penalty,
                                 delta)

            if quadratic_obj(eqp_step) <= quadratic_obj(cauchy_step):
                step = eqp_step
            else:
                step = cauchy_step
        else:
            step = cauchy_step

        model_red = linearized_red(func, func_val, deriv_val, penalty, step) - quadratic_term(step)

        next_x = x + step
        next_func_val = func.value(next_x)
        next_obj = omega(func, next_func_val, penalty)

        actual_red = curr_obj - next_obj

        red_ratio = (actual_red + stab) / (model_red + stab)

        stats.append('delta_lp', delta_lp)
        stats.append('red_ratio', red_ratio)

        step_type = "Reject"

        # Update delta
        if red_ratio >= params.rho_s:
            delta *= 2.
        else:
            lower = params.kappa_l * np.linalg.norm(step)
            upper = params.kappa_u * delta
            delta = .5 * (lower + upper)

        cauchy_norm = norm_lp(cauchy_step)

        if red_ratio >= params.rho_u:
            step_type = "Accept"

            num_accepted += 1

            next_deriv_val = func.deriv_sparse(next_x).tocsc()

            # Update Hessian based on primal diff
            if bfgs:
                primal_diff = next_x - x
                grad_diff = next_deriv_val[0, :].toarray() - deriv_val[0, :].toarray()
                grad_diff = grad_diff.ravel()
                bfgs.update(primal_diff, grad_diff)

            x = next_x
            func_val = next_func_val
            deriv_val = next_deriv_val

            lower = cauchy_norm
            upper = params.delta_lp_max if alpha == 1 else delta_lp

            delta_lp = np.clip(2. * delta_lp, lower, upper)

        else:
            num_rejected += 1
            lower = min(delta_lp, params.theta * cauchy_norm)
            upper = delta_lp

            delta_lp = .5 * (lower + upper)

        num_iterations += 1

        logger.info(f"{it:>10}|{curr_obj:20g}|{red_ratio:20g}|{step_type:>10}")

    logger.info("Finished optimization")
    logger.info(f"Final objective: {curr_obj}")

    return Result(x,
                  func_val,
                  deriv_val,
                  termination,
                  curr_obj,
                  num_iterations,
                  num_accepted,
                  num_rejected,
                  {**stats.values})
