import _mypackage

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
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


def triangle_area(xy, triangle_corner_pairs):
    return 0.5 * np.linalg.norm(*(triangle_corner_pairs - xy))


def convert_barycentric_to_cartesian(bary, corners):
    x = np.dot(bary, corners[:, 0])
    y = np.dot(bary, corners[:, 1])
    return np.array([x, y])


def test_3_model_global_mixing(legendre_expansion_orders: List[int],
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
    fig1.tight_layout()
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
        burn=500,
        steps=20_00,
        thinning=100
    )

    labels = [r'$w_1$', r'$w_2$', r'$w_3$']
    fig, ax = plt.subplots(ncols=3, nrows=4, figsize=(3 * 7, 4 * 7))
    fig.patch.set_facecolor('white')
    _, bins = np.histogram(posterior.reshape(-1, 3)[:, 0], bins=100)
    for i in range(3):
        ax[0, i].hist(posterior.reshape(-1, 3)[:, i],
                      bins=bins,
                      histtype='step',
                      density=True)
        mp.costumize_axis(ax[0, i], labels[i], "")

    print(ax.shape)
    weights = np.vstack([dirichlet(np.exp(sample)).rvs(size=1)
                         for sample in posterior.reshape(-1, len(models))])
    for j in range(1, 4):
        for i in range(3):
            if i > j - 1:
                ax[j, i].axis('off')
            elif i == j - 1:
                ax[j, i].hist(weights[:, i],
                              bins=34,
                              histtype='step',
                              density=True)
                ax[j, i].set_xlim(0, 1)
                if i != 2:
                    for t in ax[j, i].get_xticklabels():
                        t.set_fontsize(0)
                    for t in ax[j, i].get_yticklabels():
                        t.set_fontsize(0)
                else:
                    for t in ax[j, i].get_xticklabels():
                        t.set_fontsize(30)
                    for t in ax[j, i].get_yticklabels():
                        t.set_fontsize(0)
                    ax[j, i].set_xlabel(labels[i], fontsize=34)
                if i == 0:
                    ax[j, i].set_ylabel(labels[i], fontsize=34)
            else:
                ax[j, i].hist2d(weights[:, i], weights[:, j - 1], bins=34,
                                cmin=1, cmap='cividis')
                ax[j, i].set_xlim(0, 1)
                ax[j, i].set_ylim(0, 1)
                if i == 0:
                    if j == 2:
                        for t in ax[j, i].get_xticklabels():
                            t.set_fontsize(0)
                    else:
                        ax[j, i].set_xlabel(labels[i], fontsize=34)
                        for t in ax[j, i].get_xticklabels():
                            t.set_fontsize(30)
                    for t in ax[j, i].get_yticklabels():
                        t.set_fontsize(30)
                    ax[j, i].set_ylabel(labels[j - 1], fontsize=34)
                elif j == 3:
                    for t in ax[j, i].get_xticklabels():
                        t.set_fontsize(30)
                    for t in ax[j, i].get_yticklabels():
                        t.set_fontsize(0)
                    ax[j, i].set_xlabel(labels[i], fontsize=34)

    # sin30 = 0.75 ** 0.5
    # corners = np.array([[0, 0], [1, 0], [0.5, sin30]])
    # points = np.array(
    #     [
    #         [
    #             *convert_barycentric_to_cartesian(weight, corners),
    #             dirichlet.pdf(weight, np.exp(sample))
    #         ]
    #         for weight, sample in zip(weights,
    #                                   posterior.reshape(-1, len(models)))
    #     ])
    #
    # alphas = np.array([[1.2 for _ in range(3)]
    #                    for _ in range(10_000)])
    # rvs = np.array([dirichlet.rvs(alpha) for alpha in alphas]).reshape(-1, 3)
    # print(alphas.shape, rvs.shape)
    #
    # points = np.array(
    #     [
    #         [
    #             *convert_barycentric_to_cartesian(rv, corners),
    #             dirichlet.pdf(rv, alpha)
    #         ]
    #         for rv, alpha in zip(rvs, alphas)
    #     ]
    # )

    # s = ax[1, 1].scatter(points[:, 0], points[:, 1], s=5, c=points[:, 2],
    #                      cmap=mp.get_cmap(points.shape[0], 'cividis'))
    # ax[1, 1].set_aspect('equal')
    # ax[1, 1].set_xlim(0, 1)
    # ax[1, 1].set_ylim(0, sin30)
    # cax = fig.colorbar(s).ax
    # for t in cax.get_yticklabels():
    #     t.set_fontsize(19)
    fig.tight_layout()
    fig.savefig('plots/culoumb_compare_3.pdf')


# def test_n_model_global_mixing(legendre_expansion_orders: List[int],
#                                r: float,
#                                r_prime: float):
#
#     models = dict(
#         (f'{i} legendre poly', Model(order=i, r=r, r_prime=r_prime))
#         for i in legendre_expansion_orders
#     )
#     global_linear_mix = linear.LinearMixerGlobal(models=models,
#                                                  n_mix=len(models))
#     global_linear_mix.set_prior(scale=1)
#
#     def coulumb_expansion(x: float):
#         denom = np.sqrt(r ** 2 + r_prime ** 2 - 2 * r * r_prime * x)
#         return r_prime / denom
#
#     fig1, ax1 = plt.subplots(nrows=1, ncols=len(models),
#                              figsize=(len(models) * 7, 1 * 7))
#     fig1.patch.set_facecolor('white')
#     colors = mp.get_cmap(10, 'tab10')
#     xs = np.linspace(-1, 1, 100)
#     for i in range(len(models)):
#         ax1[i].plot(xs, list(models.values())[i].evaluate(xs),
#                     color=colors(i), lw=2)
#         ax1[i].plot(xs, coulumb_expansion(xs), color='black', lw=2)
#     fig1.savefig(f'plots/debug_r={r:.3f}_rp={r_prime:.3f}.pdf')
#
#     # Two ideas: 1. Feed the exact polynomial and do model mixing
#     #            2. Feed a gaussian smeered polynomial and do model mixing
#     xs = np.linspace(-1, 1, 9)
#     y_exp = np.array([coulumb_expansion(x) for x in xs])
#     y_err = np.full_like(y_exp, 0.01)
#     posterior = global_linear_mix.train(
#         y_exp=y_exp,
#         y_err=y_err,
#         model_parameters=dict((key, [xs]) for key in models.keys()),
#         steps=20_000,
#         thinning=1_000
#     )
#     print(posterior.shape)
#
#     fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(1 * 7, 7))
#     fig.patch.set_facecolor('white')
#     weights = np.vstack([dirichlet(np.exp(sample)).rvs(size=100)
#                          for sample in posterior.reshape(-1, len(models))])
#
#     colors = mp.get_cmap(10, 'tab10')
#     for i, n in enumerate(legendre_expansion_orders):
#         ax.hist(weights[:, i], color=colors(i % 10), bins=100,
#                 histtype='step', density=True)
#     mp.costumize_axis(ax, x_title=r'$w_i$',
#                       y_title='Predictive Posterior for weights')
#     fig.tight_layout()
#     fig.savefig('plots/test_legendre_polynomials_r={}_rp={}.pdf'
#                 .format(r, r_prime))


if __name__ == "__main__":
    # TODO:
    #   1. Figure out normalization, such that everything adds to 1
    #   2. Determine why the highest order does not work
    # test_n_model_global_mixing([i for i in range(2, 12, 2)], r=2.0, r_prime=1.0)
    test_3_model_global_mixing([1, 3, 10], r=1.5, r_prime=1.0)
