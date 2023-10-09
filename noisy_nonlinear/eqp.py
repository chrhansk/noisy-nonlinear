import numpy as np
import scipy.sparse
import cyipopt


from noisy_nonlinear.log import logger
from noisy_nonlinear.func import Func


class Problem(cyipopt.Problem):

    def __init__(self,
                 func,
                 func_val,
                 deriv_val,
                 hessian,
                 penalty,
                 trust_radius):

        self.func = func
        func_val = np.atleast_1d(func_val)

        (k, self.n) = deriv_val.shape

        assert scipy.sparse.issparse(deriv_val)

        assert scipy.sparse.issparse(hessian)
        assert hessian.shape == (self.n, self.n)

        self.m = k - 1

        self.cons_val = func_val[1:]

        deriv_val = deriv_val.tocsc()

        self.obj_grad = deriv_val[0, :].toarray().ravel()
        self.cons_jac = deriv_val[1:, :]

        self.cons_jac = self.cons_jac.tocoo()

        self.cl = np.concatenate([np.zeros((self.m,)),
                                  np.array([-np.inf])])

        self.cu = np.concatenate([np.zeros((self.m,)),
                                  np.array([.5 * trust_radius**2])])

        self.lb = np.concatenate([-np.inf*np.ones((self.n,)),
                                  np.zeros((self.m,)),
                                  np.zeros((self.m,))])

        self.ub = np.concatenate([np.inf * np.ones((self.n,)),
                                  np.inf * np.ones((self.m,)),
                                  np.inf * np.ones((self.m,))])

        assert (self.cl <= self.cu).all()
        assert (self.lb <= self.ub).all()

        self.penalty = penalty

        hessian = hessian.tocoo()

        hessian_data = hessian.data
        hessian_row = hessian.row
        hessian_col = hessian.col

        hessian_filter = (hessian_row >= hessian_col)

        hessian_data = hessian_data[hessian_filter]
        hessian_row = hessian_row[hessian_filter]
        hessian_col = hessian_col[hessian_filter]
        hessian_nnz = hessian_data.size

        self.lag_hess_val = np.block([[hessian_data[:, None], np.zeros((hessian_nnz, 1))],
                                      [np.zeros(self.n)[:, None], np.ones(self.n)[:, None]]])

        diag_row, diag_col = np.diag_indices(self.n)

        self.lag_hess_ind = np.block([[hessian_row[:, None], hessian_col[:, None]],
                                      [diag_row[:, None], diag_col[:, None]]])

        perm = np.lexsort(self.lag_hess_ind.T)

        self.orig_hess = hessian

        self.lag_hess_val = self.lag_hess_val[perm]
        self.lag_hess_ind = self.lag_hess_ind[perm]

        (self.lag_hess_ind, self.lag_hess_map) = np.unique(self.lag_hess_ind,
                                                           axis=0,
                                                           return_inverse=True)

        super(Problem, self).__init__(n=len(self.lb),
                                      m=len(self.cl),
                                      lb=self.lb,
                                      ub=self.ub,
                                      cl=self.cl,
                                      cu=self.cu)

        # self.add_option('derivative_test', 'second-order')
        self.add_option('print_level', 0)

    @property
    def num_problem_vars(self):
        return self.lb.size

    @property
    def num_problem_cons(self):
        return self.cl.size

    def solve(self, initial_step):

        initial_violation = self.cons_val + self.cons_jac.dot(initial_step)

        slack_upper = -np.minimum(initial_violation, 0.)
        slack_lower = np.maximum(initial_violation, 0.)

        x0 = np.concatenate([initial_step,
                             slack_upper,
                             slack_lower])

        # assert (self.lb <= x0).all()
        # assert (x0 <= self.ub).all()
        # cv = self.constraints(x0)
        # assert (self.cl <= cv).all()
        # assert (cv <= self.cu).all()

        x, info = super(Problem, self).solve(x0)

        return x[:self.n]

    def objective(self, x):
        step = x[:self.n]
        lower_slack = x[self.n:(self.n + self.m)]
        upper_slack = x[(self.n + self.m):(self.n + 2*self.m)]

        violation = np.sum(lower_slack) + np.sum(upper_slack)

        lin_obj = np.dot(step, self.obj_grad) + self.penalty*violation

        quad_obj = .5 * np.dot(step, self.orig_hess.dot(step))

        return lin_obj + quad_obj

    def gradient(self, x):
        step = x[:self.n]

        quad_grad = self.orig_hess.dot(step)

        violation_grad = np.ones((self.m,))

        grad = np.concatenate([self.obj_grad + quad_grad,
                               self.penalty * violation_grad,
                               self.penalty * violation_grad])

        num_ieq = self.func.num_ieq()

        if num_ieq > 0:
            grad[-num_ieq:] = 0.

        assert grad.shape == (self.num_problem_vars,)

        return grad

    def constraints(self, x):
        step = x[:self.n]
        upper_slack = x[self.n:(self.n + self.m)]
        lower_slack = x[(self.n + self.m):(self.n + 2*self.m)]

        step_product = self.cons_jac.dot(step)

        violation = self.cons_val + step_product - upper_slack + lower_slack

        step_normsq = np.dot(step, step)

        cons_val = np.concatenate([violation, np.array([.5 * step_normsq])])

        assert cons_val.shape == (self.num_problem_cons,)

        return cons_val

    def jacobianstructure(self):
        rj = self.cons_jac.row
        cj = self.cons_jac.col

        (rm, cm) = np.diag_indices(self.m)
        (rs, cs) = Func.full_indices((1, self.n))

        rows = np.concatenate([rj, rm, rm, rs + self.m])
        cols = np.concatenate([cj, cm + self.n, cm + self.n + self.m, cs])

        assert (rows >= 0).all()
        assert (rows < self.num_problem_cons).all()

        assert (cols >= 0).all()
        assert (cols < self.num_problem_vars).all()

        return rows, cols

    def _num_vars(self):
        return self.n + 2*self.m

    def _num_cons(self):
        return self.m + 1

    def _jacobian_sparse(self, x):
        rows, cols = self.jacobianstructure()
        data = self.jacobian(x)

        return scipy.sparse.coo_matrix((data, (rows, cols)),
                                       shape=(self._num_cons(), self._num_vars()))

    def _jacobian_dense(self, x):
        step = x[:self.n]

        blocks = [[self.cons_jac.toarray(), np.eye(self.m), -np.eye(self.m)],
                  [step, np.zeros(self.m), np.zeros(self.m)]]

        return np.block(blocks)

    def jacobian(self, x):
        step = x[:self.n]

        values = [self.cons_jac.data,
                  np.full(self.m, -1.),
                  np.ones(self.m),
                  step]

        return np.concatenate(values)

    def hessianstructure(self):
        rows = self.lag_hess_ind[:, 0]
        cols = self.lag_hess_ind[:, 1]

        assert rows.shape == cols.shape
        assert rows.ndim == 1

        assert (rows >= 0).all()
        assert (rows < self.num_problem_vars).all()

        assert (cols >= 0).all()
        assert (cols < self.num_problem_vars).all()

        assert (rows >= cols).all()

        return rows, cols

    def hessian(self, x, lagrange, obj_factor):
        lag = lagrange[self.m]

        (hess_nnz, _) = self.lag_hess_ind.shape

        hess_val = self.lag_hess_val.dot(np.array([obj_factor, lag]))

        combined_hess_val = np.zeros((hess_nnz,))

        for (i, j) in enumerate(self.lag_hess_map):
            combined_hess_val[j] += hess_val[i]

        assert combined_hess_val.shape == self.lag_hess_ind[:, 0].shape

        return combined_hess_val


def solve_eqp(func,
              initial_step,
              func_val,
              deriv_val,
              hessian,
              penalty,
              trust_radius):

    problem = Problem(func,
                      func_val,
                      deriv_val,
                      hessian,
                      penalty,
                      trust_radius)

    num_vars = problem.num_problem_vars
    num_cons = problem.num_problem_cons

    logger.debug(f"Starting to solve EQP with {num_vars} variables and {num_cons} constraints")

    sol = problem.solve(initial_step)

    logger.debug("Finished solving EQP")

    return sol
