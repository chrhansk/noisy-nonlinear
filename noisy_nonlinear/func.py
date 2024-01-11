import numpy as np
import scipy.sparse

from noisy_nonlinear.sampling import sample_ball

default_seed = 42


def pos(x):
    return np.maximum(x, 0.)


def sum_pos(x):
    return np.sum(np.maximum(x, 0.))


def one_norm(x):
    return np.linalg.norm(x, ord=1)


def feas_res(func, func_val):
    func_val = np.atleast_1d(func_val)
    cons = func_val[1:]

    num_ieq = func.num_ieq()

    viol_eq = np.absolute(cons[:-num_ieq])
    viol_ieq = pos(cons[-num_ieq:])

    max_viol_eq = viol_eq.max()
    max_viol_ieq = viol_ieq.max()

    return max(max_viol_eq, max_viol_ieq)


# Non-smooth outer function, containing 1-norm
def omega(func, func_val, penalty):
    func_val = np.atleast_1d(func_val)
    obj = func_val[0]
    cons = func_val[1:]

    num_ieq = func.num_ieq()

    viol_eq = np.absolute(cons[:-num_ieq])
    viol_ieq = pos(cons[-num_ieq:])

    return obj + penalty * (np.sum(viol_eq) + np.sum(viol_ieq))


def linearized_obj(func, func_val, deriv_val, penalty, step):
    next_val = func_val + deriv_val.dot(step)
    return omega(func, next_val, penalty)


def linearized_red(func, func_val, deriv_val, penalty, step):
    step_prod = deriv_val.dot(step)

    obj_reduction = -step_prod[0]

    cons_val = func_val[1:]
    cons_step_prod = step_prod[1:]

    # unconstrained
    if cons_val.size == 0:
        return obj_reduction

    next_cons_val = cons_val + cons_step_prod

    num_ieq = func.num_ieq()

    norm_diff = 0

    if num_ieq > 0:
        cons_val_eq = cons_val[:-num_ieq]
        cons_val_ieq = cons_val[-num_ieq:]

        next_cons_val_eq = next_cons_val[:-num_ieq]
        next_cons_val_ieq = next_cons_val[-num_ieq:]

        norm_diff = one_norm(cons_val_eq) - one_norm(next_cons_val_eq)
        norm_diff += sum_pos(cons_val_ieq) - sum_pos(next_cons_val_ieq)
    else:
        norm_diff = one_norm(cons_val) - one_norm(next_cons_val)

    return obj_reduction + penalty * norm_diff


class Func:
    def value(self, x):
        pass

    def opt(self):
        pass

    @staticmethod
    def full_indices(shape):
        r, c = np.indices(shape)
        return r.flatten(), c.flatten()

    def deriv_values(self, x):
        pass

    def deriv_struct(self):
        pass

    def deriv_dense(self, x):
        num_cols = self.dim()
        num_rows = 1 + self.num_cons()

        deriv = np.zeros((num_rows, num_cols))

        (r, c) = self.deriv_struct()

        deriv[r, c] = self.deriv_values(x)

        return deriv

    def deriv_sparse(self, x, **kwds):
        num_cols = self.dim()
        num_rows = 1 + self.num_cons()

        [r, c] = self.deriv_struct()

        v = self.deriv_values(x)

        return scipy.sparse.coo_matrix((v, (r, c)),
                                       shape=(num_rows, num_cols))

    def hess_values(self, x):
        return np.ones(self.dim())

    def hess_struct(self):
        return np.diag_indices(self.dim())

    def hess_sparse(self, x):
        num_cols = self.dim()

        (r, c) = self.hess_struct()
        v = self.hess_values(x)

        return scipy.sparse.coo_matrix((v, (r, c)),
                                       shape=(num_cols, num_cols))

    def hess_dense(self, x):
        num_cols = self.dim()

        (r, c) = self.hess_struct()
        v = self.hess_values(x)

        hess = np.zeros((num_cols, num_cols))

        hess[r, c] = v

        return hess

    def value_error(self):
        return 0.

    def deriv_error(self):
        return 0.

    def deriv_lipschitz(self, pmin, pmax):
        raise NotImplementedError

    def num_cons(self):
        pass

    # Layout: (eq, ieq)
    def num_ieq(self):
        return 0

    def dim(self):
        pass

    def is_noisy(self):
        return (self.value_error() > 0.) or (self.deriv_error() > 0.)

    def orig_func(self):
        return self


class NoisyFunc(Func):

    def __init__(self, func, value_error, deriv_error, **kwds):
        assert value_error >= 0.
        assert deriv_error >= 0.

        super(NoisyFunc, self).__init__()

        self._value_error = value_error
        self._deriv_error = deriv_error
        self._func = func

        seed = kwds.get('seed', default_seed)

        self._random = np.random.RandomState(seed=seed)

    def orig_func(self):
        return self._func

    def value_error(self):
        return self._value_error

    def deriv_error(self):
        return self._deriv_error

    def _add_noise(self, value, error_bound):
        value = np.atleast_1d(value)
        dim = value.size

        noise = sample_ball(self._random, dim, error_bound)

        norm = np.linalg.norm(noise, ord=2)

        assert (norm <= error_bound) or np.allclose(norm, error_bound)

        noise = noise.reshape(value.shape)

        return value + noise

    def value(self, x):
        value = self._func.value(x)

        return self._add_noise(value,
                               self.value_error())

    def opt(self):
        return self._func.opt()

    def deriv_values(self, v):
        deriv = self._func.deriv_dense(v)
        noisy_deriv = self._add_noise(deriv, self.deriv_error())

        return noisy_deriv[self.deriv_struct()]

    def deriv_struct(self):
        return Func.full_indices((1 + self.num_cons(), self.dim()))

    def deriv_lipschitz(self, pmin, pmax):
        return self._func.deriv_lipschitz(pmin, pmax)

    def hess_values(self, x):
        return self._func.hess_values(x)

    def hess_struct(self):
        return self._func.hess_struct()

    def num_cons(self):
        return self._func.num_cons()

    def num_ieq(self):
        return self._func.num_ieq()

    def dim(self):
        return self._func.dim()
