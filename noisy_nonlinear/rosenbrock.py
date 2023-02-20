import numpy as np

from noisy_nonlinear.func import Func


class Rosenbrock(Func):

    x_opt = np.array([1., 1.])

    a = 1
    b = 100

    def value(self, v):
        [x, y] = v

        obj = (self.a - x)**2 + self.b*(y - x**2)**2

        return np.concatenate([np.array([obj]), x - Rosenbrock.x_opt])

    def opt(self):
        return self.x_opt

    def grad(self, v):
        [x, y] = v
        return np.array([4*(x**2 - y)*self.b*x - 2*self.a + 2*x,
                         -2*(x**2 - y)*self.b])

    def deriv_values(self, v):
        return np.concatenate([self.grad(v),
                               np.ones(self.num_cons())])

    def deriv_struct(self):
        rgrad, cgrad = Func.full_indices((1, self.dim()))
        rcons, ccons = np.diag_indices(self.num_cons())

        r = np.concatenate([rgrad, rcons + 1])
        c = np.concatenate([cgrad, ccons])

        return (r, c)

    def value_error(self):
        return 0.

    def deriv_error(self):
        return 0.

    def deriv_lipschitz(self, pmin, pmax):
        xmin, ymin = pmin
        xmax, ymax = pmax

        b = self.b

        def trace_bound(x, y):
            return 8*b*x**2 + 4*(x**2 - y)*b + 2*b + 2

        max_trace = 0.

        for x in [xmin, xmax]:
            for y in [ymin, ymax]:
                max_trace = max(max_trace, trace_bound(x, y))

        return max_trace

    def hess_values(self, v):
        [x, y] = v

        b = self.b

        hess = np.array([[8*b*x**2 + 4*(x**2 - y)*b + 2, -4*b*x],
                         [-4*b*x, 2*b]])

        return hess.ravel()

    def hess_struct(self):
        d = self.dim()
        return Func.full_indices((d, d))

    def num_cons(self):
        return 2

    def dim(self):
        return 2
