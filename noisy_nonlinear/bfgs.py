import numpy as np
import scipy.sparse

from noisy_nonlinear.log import logger


class NoisyBFGS:

    damping_factor = 0.2

    def __init__(self, func, params):
        self.params = params
        self.func = func

        val = min(self._max_scale(), 1.)

        self._has_initial_scale = False
        self.matrix = val * np.eye(func.dim())

    def _max_scale(self):
        return self.params.beta / self.func.dim()

    def get_matrix(self):
        return scipy.sparse.coo_matrix(self.matrix)

    def _set_initial_scale(self, primal_step, grad_step):
        grad_step_dot = np.dot(primal_step, grad_step)
        grad_dot = np.dot(grad_step, grad_step)

        if grad_step_dot > 0.:
            val = grad_dot / grad_step_dot
            val = min(self._max_scale(), val)
            self.matrix = val * np.eye(self.func.dim())

        self._has_initial_scale = True

    def update(self, primal_step, grad_step):
        logger.debug("Performing BFGS update")

        if not(self._has_initial_scale):
            self._set_initial_scale(primal_step, grad_step)

        primal_prod = np.dot(self.matrix, primal_step)

        inner_dot = np.dot(primal_step, grad_step)

        bidir_dot = np.dot(primal_step, primal_prod)

        assert bidir_dot != 0., "Zero step"
        assert inner_dot != 0., "On zero hyperplan"

        # Damping
        if inner_dot >= self.damping_factor * bidir_dot:
            pass
        else:
            logger.debug("Damping BFGS update")
            factor = (1. - self.damping_factor) * bidir_dot / (bidir_dot - inner_dot)
            grad_step = factor * grad_step + (1. - factor) * primal_prod
            inner_dot = factor * inner_dot + (1. - factor) * inner_dot

        next_hessian = np.copy(self.matrix)
        next_hessian -= 1./bidir_dot * np.outer(primal_prod, primal_prod)
        next_hessian += 1./inner_dot * np.outer(grad_step, grad_step)

        next_trace = np.trace(next_hessian)
        # Skip update if trace (upper bound on norm) becomes too large
        if next_trace > self.params.beta:
            logger.debug(f"Skipping BFGS update (next trace: {next_trace})")
        else:
            self.matrix = next_hessian
