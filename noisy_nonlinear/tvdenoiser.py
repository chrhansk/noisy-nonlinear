import numpy as np
import scipy.sparse

from noisy_nonlinear.denoiser import Denoiser


def cartesian(X, Y):
    return np.transpose([np.tile(X, len(Y)),
                         np.repeat(Y, len(X))])


def tvmat(m, n):
    I = np.arange(m)
    J = np.arange(n)

    indices = np.concatenate([cartesian(I[1:], J),
                              cartesian(I[:-1], J),
                              cartesian(I, J[1:]),
                              cartesian(I, J[:-1])])

    values = np.concatenate([np.full(((m - 1)*n,), 1.),
                             np.full(((m - 1)*n,), -1.),
                             np.full((m*(n - 1),), 1.),
                             np.full((m*(n - 1),), -1.)])

    cols = indices.dot(np.array([m, 1]))

    rows = np.concatenate([np.arange((m - 1)*n),
                           np.arange((m - 1)*n),
                           np.arange(m*(n - 1)) + (m - 1)*n,
                           np.arange(m*(n - 1)) + (m - 1)*n])

    perm = np.lexsort((cols, rows))

    return scipy.sparse.coo_matrix((values[perm],
                                    (rows[perm], cols[perm])))


class TVDenoiser(Denoiser):
    def __init__(self, filename, noise_level=.1):
        super(TVDenoiser, self).__init__(filename, noise_level)

        (self.m, self.n) = self.image.shape

        self.mat = tvmat(self.m, self.n)

    def value(self, x):
        obj = super(TVDenoiser, self).value(x)

        prod = self.mat.dot(x)

        return np.concatenate([obj, prod])

    def deriv_values(self, x):
        obj_deriv = super(TVDenoiser, self).deriv_values(x)

        values = np.concatenate([obj_deriv, self.mat.data])

        return values

    def deriv_struct(self):
        obj_row, obj_col = super(TVDenoiser, self).deriv_struct()

        mat_row = self.mat.row
        mat_col = self.mat.col

        row = np.concatenate([obj_row, 1 + mat_row])
        col = np.concatenate([obj_col, mat_col])

        return row, col

    def fidelity(self, x):
        return super(TVDenoiser, self).value(x).item()

    def num_cons(self):
        m = self.m
        n = self.n

        return m*(n - 1) + n*(m - 1)

    def orig_func(self):
        if self.noise_level == 0.:
            return self
        return TVDenoiser(self.filename, noise_level=0.)


if __name__ == "__main__":
    from .noisy_solver import solve
    import logging

    logging.basicConfig(level=logging.DEBUG)

    filename = "image_8.png"

    noise_level = 0.1
    func = TVDenoiser(filename, noise_level)

    x_0 = np.zeros((func.dim(),))

    penalty = 0.1

    res_stab = solve(func, penalty, x_0, solve_stab=True)
