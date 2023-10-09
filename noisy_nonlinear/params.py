import dataclasses
from dataclasses import dataclass


@dataclass
class Params:
    rho_u: float = .1
    rho_s: float = .5
    kappa_l: float = .1
    kappa_u: float = .8
    theta: float = .5
    cauchy_eta: float = 0.1
    cauchy_tau: float = 0.5
    delta_lp_max: float = 10.
    beta: float = 100.

    delta_0: float = 1.
    delta_lp_0: float = 1.

    max_it: int = 50

    perform_eqp: bool = True

    stabilization: float = None

    use_quasi_newton: bool = False

    collect_stats: bool = True

    deriv_check: bool = False
    deriv_pert: float = 1e-8
    deriv_tol: float = 1e-6

    def __post_init__(self):
        self.validate()

    def validate(self):
        assert 0 < self.rho_u < self.rho_s < 1
        assert 0 < self.kappa_l <= self.kappa_u < 1
        assert self.theta > 0.
        assert 0 < self.cauchy_eta < 1.
        assert 0 < self.cauchy_tau < 1.
        assert self.delta_lp_0 <= self.delta_lp_max
        assert self.delta_lp_max >= 1.
        assert self.max_it > 0
        assert 0 <= self.beta

    def is_stabilized(self):
        return self.stabilization != 0.

    def replace(self, **values):
        return dataclasses.replace(self, **values)
