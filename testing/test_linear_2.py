import _mypackage

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import dirichlet
from scipy.special import legendre
from typing import List

from Taweret.core.base_model import BaseModel
from Taweret.mix import linear

import my_plotting as mp


class Model(BaseModel):
    def __init__(self, order, r, r_prime):
        self.order = order
        self.r = r
        self.r_prime = r_prime

    def evaluate(self, cos_theta):
        return_value = 0
        for n in range(self.order):
            value = (self.r_prime / self.r) ** (n + 1)
            value *= legendre(n)(cos_theta)
            return_value += value
        return return_value

    def log_likelihood_elementwise(self, y_exp, y_err, cos_theta):
        return_value = np.exp(
            -(self.evaluate(cos_theta) - y_exp) ** 2 / 2 * y_err ** 2
        )
        return_value /= np.sqrt(2 * np.pi * y_err ** 2)
        return np.prod(return_value)

    def set_prior(self):
        pass


def test_n_model_global_mixing(legendre_expansion_orders: List[int],
                               r: float,
                               r_prime: float):

    models = dict(
        (f'{i} legendre poly', Model(order=i, r=r, r_prime=r_prime))
        for i in legendre_expansion_orders
    )
    global_linear_mix = linear.LinearMixerGlobal(models=models,
                                                 n_mix=len(models))
    global_linear_mix.set_prior(scale=1)

    def coulumb_expansion(x: float):
        denom = np.sqrt(r ** 2 + r_prime ** 2 - 2 * r * r_prime * x)
        return r_prime / denom

    fig1, ax1 = plt.subplots(nrows=1, ncols=len(models),
                             figsize=(len(models) * 7, 1 * 7))
    fig1.patch.set_facecolor('white')
    colors = mp.get_cmap(10, 'tab10')
    xs = np.linspace(-1, 1, 100)
    for i in range(len(models)):
        ax1[i].plot(xs, list(models.values())[i].evaluate(xs),
                    color=colors(i), lw=2)
        ax1[i].plot(xs, coulumb_expansion(xs), color='black', lw=2)
    fig1.savefig(f'plots/debug_r={r:.3f}_rp={r_prime:.3f}.pdf')

    # Two ideas: 1. Feed the exact polynomial and do model mixing
    #            2. Feed a gaussian smeered polynomial and do model mixing
    xs = np.linspace(-1, 1, 9)
    y_exp = np.array([coulumb_expansion(x) for x in xs])
    y_err = np.full_like(y_exp, 0.01)
    posterior = global_linear_mix.train(
        y_exp=y_exp,
        y_err=y_err,
        model_parameters=dict((key, [xs]) for key in models.keys()),
        steps=20_000,
        thinning=1_000
    )
    print(posterior.shape)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(1 * 7, 7))
    fig.patch.set_facecolor('white')
    weights = np.vstack([dirichlet(np.exp(sample)).rvs(size=100)
                         for sample in posterior.reshape(-1, len(models))])

    colors = mp.get_cmap(10, 'tab10')
    for i, n in enumerate(legendre_expansion_orders):
        ax.hist(weights[:, i], color=colors(i % 10), bins=100,
                histtype='step', density=True)
    mp.costumize_axis(ax, x_title=r'$w_i$',
                      y_title='Predictive Posterior for weights')
    fig.tight_layout()
    fig.savefig('plots/test_legendre_polynomials_r={}_rp={}.pdf'
                .format(r, r_prime))


if __name__ == "__main__":
    # TODO:
    #   1. Figure out normalization, such that everything adds to 1
    #   2. Determine why the highest order does not work
    # test_n_model_global_mixing([i for i in range(2, 12, 2)], r=2.0, r_prime=1.0)
    test_n_model_global_mixing([1, 5, 10], r=1.5, r_prime=1.0)
