import numpy as np
import scipy as sp
import scipy.optimize
import scipy.sparse

from noisy_nonlinear.log import logger


def solve_lp(func, func_val, deriv_val, penalty, trust_radius):
    func_val = np.atleast_1d(func_val)

    assert scipy.sparse.issparse(deriv_val)

    (k, n) = deriv_val.shape

    assert func_val.shape == (k,)

    deriv_val = deriv_val.tocsc()

    grad = deriv_val[0, :].toarray().ravel()
    jac = deriv_val[1:, :]

    jac = scipy.sparse.coo_matrix(jac)

    cons = func_val[1:]

    m = k - 1

    ident = scipy.sparse.eye(m)

    A_eq = scipy.sparse.hstack([jac, -ident, ident])
    b_eq = -cons

    c = np.concatenate([grad,
                        penalty*np.ones((m,)),
                        penalty*np.ones(m,)])

    # Variables corresponding to a negative violation
    # of an inequality should *not* be penalized
    num_ieq = func.num_ieq()

    if num_ieq > 0:
        c[-num_ieq:] = 0.

    xmin = np.concatenate([-trust_radius * np.ones((n,)),
                           np.zeros(m,),
                           np.zeros(m,)])

    xmax = np.concatenate([trust_radius * np.ones(n,),
                           np.inf * np.ones(m,),
                           np.inf * np.ones(m,)])

    bounds = np.vstack([xmin, xmax]).T

    [num_cols] = xmin.shape
    [num_rows] = b_eq.shape

    logger.debug(f"Starting to solve LP with {num_cols} variables and {num_rows} constraints")

    res = sp.optimize.linprog(c,
                              A_eq=A_eq,
                              b_eq=b_eq,
                              bounds=bounds,
                              method='highs')

    logger.debug("Finished solving LP")

    assert res.success

    return (res.x[:n], res.fun)
