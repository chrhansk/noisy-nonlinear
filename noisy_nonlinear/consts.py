import numpy as np


# Lipschitz constant of omega function
def const_lip_omega(penalty, num_cons):
    return np.sqrt(1 + (penalty * penalty) * num_cons)


# Factor for conversion
# between LP- and 2-norm
def const_gamma(dim):
    return np.sqrt(dim)


def const_A(params,
            dim):
    gamma = const_gamma(dim)

    return min(params.theta * gamma,
               params.delta_0,
               params.delta_lp_0 * gamma,
               gamma / params.delta_lp_max)


def const_m_eps_0(func, lip_omega):
    return 2. * lip_omega * func.value_error()


def const_m_eps_1(func, lip_omega):
    return lip_omega * func.deriv_error()


# Problem: Need Lipschitz constant of derivative
def const_m_eps_2(lip_deriv, lip_omega, lip_hessian):
    return lip_omega * lip_deriv + 0.5 * lip_hessian


def stabilization(func, penalty, params):
    lip_omega = const_lip_omega(penalty, func.num_cons())
    m_eps_0 = const_m_eps_0(func, lip_omega)
    m_eps_1 = const_m_eps_1(func, lip_omega)

    return (m_eps_0 + m_eps_1) / (1. - params.rho_u)


def const_B(func,
            penalty,
            lip_deriv,
            params):
    gamma = const_gamma(func.dim())
    lip_omega = const_lip_omega(penalty, func.num_cons())
    m_eps_2 = const_m_eps_2(lip_deriv, lip_omega, params.beta)

    l_lin = gamma*lip_omega*(lip_deriv + func.value_error())

    min_1 = (1 - params.rho_u) * params.cauchy_eta / \
        (gamma * m_eps_2 * params.delta_lp_max) * \
        min(params.theta**2, params.kappa_u**2)

    min_2 = gamma / l_lin

    min_3 = 2*(1 - params.cauchy_eta)*params.cauchy_tau / \
        (params.beta*gamma*params.delta_lp_max)

    return min(min_1, min_2, min_3)


def const_delta(func,
                penalty,
                lip_deriv,
                params):
    gamma = const_gamma(func.dim())

    A = const_A(params, func.dim())
    B = const_B(func, penalty, lip_deriv, params)

    common_factor = stabilization(func, penalty, params)*(1 - params.rho_u) \
        * gamma * params.delta_lp_max / (params.rho_u * params.cauchy_eta)

    max_1 = np.sqrt(common_factor / B)
    max_2 = common_factor / A

    return max(max_1, max_2)
