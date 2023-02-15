#
# File: dirichlet_prior.py
# Author: Kevin Ingles
# Description: Helper classes that define a useful representation of the 
#              Dirichlet distribution that is consumed by the bilby module

import bilby
import numpy as np
from scipy.special import erfinv 
from scipy.stats import dirichlet
from typing import Optional, Union


# ------------------------------------------------------------------------
class DirichletPriorElement(bilby.core.prior.Prior):
    def __init__(
            self,
            n_dim: int,
            order: int,
            name: Optional[str] = None,
            latex_label: Optional[str] = None,
    ):
        super().__init__(name=name, latex_label=latex_label)
        self.n_dim = n_dim
        self.order = order
        self.minimum = 0        # These should be constraints on the
        self.maximum = np.inf   # shape parameters

    def rescale(
            self,
            val: float
    ) -> float:
        mu = 0
        scale = 1
        return np.exp(mu + np.sqr(2) * scale * erfinv(2 * val - 1))

    def prob(
            self,
            val: Union[float, int, np.ndarray]
    ) -> Union[float, np.ndarray]:
        if np.any(val < self.minimum):
            _prob = 0
        else:
            samples = dirichlet(val).rvs()
            _prob = dirichlet(val).pdf(samples)
        return _prob

    def ln_prob(
            self,
            val: Union[float, int, np.ndarray]
    ) -> Union[float, np.ndarray]:
        if np.any(val < self.minimum):
            _ln_prob = -np.inf
        else:
            samples = dirichlet(val).rvs()
            _ln_prob = dirichlet(val).logpdf(samples)
        return _ln_prob

# ------------------------------------------------------------------------
