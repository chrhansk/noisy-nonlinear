import numpy as np

from noisy_nonlinear.func import Func


class Quad(Func):
    """
    Quadratic example from noisy trust region paper
    """

    default_diag = 10**np.arange(-5, -3.25 + 0.01, 0.25)

    def __init__(self, diag=None):
        if diag is None:
            diag = self.default_diag
        self.diag = np.copy(diag)

        assert (self.diag > 0).all()

    def value(self, v):
        return np.append(np.array(.5 * np.dot(v, self.diag*v)), [v])

    def opt(self):
        return np.zeros((self.dim(),))

    def deriv_values(self, x):
        return np.concatenate([self.diag * x,
                               np.ones(self.num_cons())])

    def deriv_struct(self):
        rgrad, cgrad = Func.full_indices((1, self.dim()))
        rcons, ccons = np.diag_indices(self.num_cons())

        r = np.concatenate([rgrad, rcons + 1])
        c = np.concatenate([cgrad, ccons])

        return (r, c)

    def hess_values(self, x):
        return self.diag

    def hess_struct(self):
        return np.diag_indices(self.dim())

    def deriv_lipschitz(self, pmin, pmax):
        return self.diag.max()

    def num_cons(self):
        return self.dim()

    def dim(self):
        return self.diag.size


if __name__ == "__main__":

    func = Quad()

    x = np.ones_like(func.diag)

    print(func.deriv_dense(x))
    print(func.deriv_sparse(x))

    pass
