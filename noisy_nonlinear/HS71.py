import numpy as np

from noisy_nonlinear.func import Func


def sq(x):
    return x*x


class HS71(Func):

    x_0 = np.array([1.,
                    5.,
                    5.,
                    1.])

    x_opt = np.array([1.,
                      4.742999,
                      3.821151,
                      1.379408])

    def value(self, x):
        value = np.concatenate([np.array([self.obj(x)]),
                                self.cons(x)])

        return value

    def obj(self, x):
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def cons(self, x):
        c1 = np.sum(x*x) - 40.
        c2 = -(x.prod()) + 25

        vlower = -x + 1.
        vupper = x - 5.

        return np.concatenate([np.array([c1]),
                               np.array([c2]),
                               vlower,
                               vupper])

    def opt(self):
        return self.x_opt

    def grad_values(self, x):
        return np.array([(x[0] + x[1] + x[2])*x[3] + x[0]*x[3],
                         x[0]*x[3],
                         x[0]*x[3] + 1,
                         (x[0] + x[1] + x[2])*x[0]])

    def cons_jac_values(self, x):
        c1 = 2. * x
        c2 = np.array([-x[1]*x[2]*x[3],
                       -x[0]*x[2]*x[3],
                       -x[0]*x[1]*x[3],
                       -x[0]*x[1]*x[2]])

        return np.concatenate([c1,
                               c2,
                               -np.ones(self.dim()),
                               np.ones(self.dim())])

    def deriv_values(self, x):
        values = np.concatenate([self.grad_values(x),
                                 self.cons_jac_values(x)])

        return values

    def deriv_struct(self):
        rfull, cfull = Func.full_indices((1, self.dim()))

        rdiag, cdiag = np.diag_indices(self.dim())

        rows = np.concatenate([rfull,
                               rfull + 1,
                               rfull + 2,
                               rdiag + 3,
                               rdiag + 3 + self.dim()])

        cols = np.concatenate([cfull,
                               cfull,
                               cfull,
                               cdiag,
                               cdiag])

        return (rows, cols)

    # def hess_values(self, v)

    # def hess_struct(self)


    # everything is an inequality except
    # for the first equation
    def num_ieq(self):
        return self.num_cons() - 1

    def hess_values(self, x):
        return np.array([])

    def hess_struct(self):
        return (np.array([], dtype=int),
                np.array([], dtype=int))

    def num_cons(self):
        return 10

    def dim(self):
        return 4
