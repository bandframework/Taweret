import _mypackage

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import dirichlet

from Taweret.core.base_model import BaseModel
from Taweret.mix import linear

import my_plotting as mp

# TODO: Move to notebook and upload to Tawere documentatio


class Model1(BaseModel):
    def evaluate(self):
        return 10.0

    def log_likelihood_elementwise(self, y_exp, y_err):
        result = np.exp(-(10 - y_exp) ** 2 / (2 * y_err ** 2))
        result /= np.sqrt(2 * np.pi * y_err ** 2)
        return result

    def set_prior(self):
        pass


class Model2(BaseModel):
    def evaluate(self):
        return -10.0

    def log_likelihood_elementwise(self, y_exp, y_err):
        result = np.exp(-(-10 - y_exp) ** 2 / (2 * y_err ** 2))
        result /= np.sqrt(2 * np.pi * y_err ** 2)
        return result

    def set_prior(self):
        pass


def test_two_model_global_mixing(loc):
    models = {'plus': Model1(), 'minus': Model2()}
    global_linear_mix = linear.LinearMixerGlobal(models=models,
                                                 n_mix=len(models))
    global_linear_mix.set_prior(scale=1)

    y_exp = loc
    y_err = 0.05
    posterior = global_linear_mix.train(y_exp=y_exp,
                                        y_err=y_err)

    cols = 2
    rows = 1
    fig, ax = plt.subplots(ncols=cols, nrows=rows, figsize=(cols * 7, rows * 7))
    fig.patch.set_facecolor('white')
    weights = np.vstack([dirichlet(np.exp(sample)).rvs(size=100)
                         for sample in posterior.reshape(-1, len(models))])

    # The conclusion to much trial and error is that it does not make sense to
    # try and plot a 2D Dir distribution in 3 dimensions.
    # It is much more useful to plot a single histogram of 1 weight sample, as
    # that will automatically determine that of the other.
    # However, this does mean things will no be properly normalizedd
    #
    # Another option is to take the the MAP of the hyperparameter posterior
    # to generate the distribution of the weights and just plot that
    bins = ax[0].hist(weights[:, 0], color='red', bins=100,
                      histtype='step', label=r'$w_1$')[1]
    ax[0].hist(weights[:, 1], color='blue', bins=bins,
               histtype='step', label=r'$w_2$')
    ax[0].legend(fontsize=20)
    mp.costumize_axis(ax[0], x_title=r'$w_i$',
                      y_title=r'$\mathcal P_\mathrm{pred}(w_i| \alpha)$')

    hyper_parameters = np.array([np.exp(sample)
                                 for sample in posterior.reshape(-1,
                                                                 len(models))])
    n_alpha_1, bin_edges = ax[1].hist(
        hyper_parameters[:, 0],
        color='black', bins=100,
        density=True,
        histtype='step',
        range=(1, 15),
        label=r'$\alpha_1$'
    )[0:2]
    n_alpha_2 = ax[1].hist(
        hyper_parameters[:, 1],
        color='gray',
        bins=bin_edges,
        density=True,
        histtype='step',
        label=r'$\alpha_2$')[0]
    ax[1].legend(fontsize=20)
    mp.costumize_axis(ax[1], x_title=r'$\alpha$',
                      y_title=r'$\mathcal P_\mathrm{pred}(\alpha)$')

    fig.tight_layout()
    fig.savefig(f'plots/2_model_comp_loc={loc}.pdf')

    fig2, ax2 = plt.subplots(ncols=2, nrows=2, figsize=(2 * 7, 2 * 7))
    fig2.patch.set_facecolor('white')
    n_samples = posterior.reshape(-1, len(models)).shape[0]
    ax2[0, 0].plot(np.arange(n_samples), posterior.reshape(-1, len(models))[:, 0],
                   color='red')
    ax2[1, 0].plot(np.arange(n_samples), posterior.reshape(-1, len(models))[:, 1],
                   color='blue')
    mp.costumize_axis(ax2[0, 0], x_title='steps',
                      y_title=r'Trace for hyperparameter of $w_1$')
    mp.costumize_axis(ax2[1, 0], x_title='steps',
                      y_title=r'Trace for hyperparameter of $w_2$')

    def autocorrelation(chain, max_lag=100):
        auto_correlation = np.empty(max_lag + 1)
        chain1d = chain - np.mean(chain)
        for lag in range(max_lag + 1):
            shifted = chain[lag:]
            if lag == 0:
                unshifted = chain1d
            else:
                unshifted = chain1d[:-lag]
            normalization = np.sqrt(np.dot(unshifted, unshifted))
            normalization *= np.sqrt(np.dot(shifted, shifted))
            auto_correlation[lag] = np.dot(unshifted, shifted) / normalization
        return auto_correlation

    ax2[0, 1].plot(autocorrelation(posterior.reshape(-1, len(models))[:, 0]),
                   color='red')
    ax2[1, 1].plot(autocorrelation(posterior.reshape(-1, len(models))[:, 1]),
                   color='blue')
    mp.costumize_axis(ax2[0, 1], x_title='',
                      y_title=r'Autocorrelation of hyperparameter of $w_1$')
    mp.costumize_axis(ax2[1, 1], x_title='',
                      y_title=r'Autocorrelation of hyperparameter of $w_2$')
    fig2.tight_layout()
    fig2.savefig(f'plots/2_model_comp_tace_and_autocorrelation_loc={loc}.pdf')


if __name__ == "__main__":
    test_two_model_global_mixing(loc=10)
    test_two_model_global_mixing(loc=0)
    test_two_model_global_mixing(loc=-10)
