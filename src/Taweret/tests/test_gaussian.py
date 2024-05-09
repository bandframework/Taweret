###########################################################
# Testing file for gaussian.py (SAMBA multivariate BMM)
# we can test both the samba_models and gaussian.py code
# in this pytest script here
# Author: Alexandra Semposki
# Date: 03 April 2023
###########################################################

import sys
from pathlib import Path

# Set Taweret Path
dirname = Path(__file__).absolute()
cwd = dirname.parent
taweret_wd = str(dirname).split("test")[0]
sys.path.append(taweret_wd)

from Taweret.core.base_model import *
from Taweret.core.base_mixer import *
from Taweret.mix.gaussian import *
from Taweret.models.samba_models import *
import pytest
import numpy as np

# import sys
# sys.path.append('../../Taweret')


# set up the order to test at
order = 3

# List models to mix (form dict)
model_1 = Loworder(order=order)
model_2 = Highorder(order=order)

labels = [r'$N_s = 3$', r'$N_l = 3$']

models = {
    "1": model_1,
    "2": model_2
}

# Evaluate the models at the input points
g = np.linspace(1e-6, 1.0, 100)
predict = []

for i in models.keys():
    predict.append(models[i].evaluate(g))

# call the Multivariate class
n_models = 2
mixed = Multivariate(g, models, n_models=n_models)

# call the predict function to construct the PPD
ci = 68
posterior_draws, mixed_mean, mixed_intervals, std_dev = mixed.predict(ci=ci)

# now call for evaluation of the weights
weights = mixed.evaluate_weights()

###########################################################
# testing suite
###########################################################


def test_models():

    # individual models
    assert model_1 is not None, "model_1 is None"
    assert model_2 is not None, "model_2 is None"

    # dict of models
    assert models is not None, "dict of models is not filled"
    assert np.array_equal(np.asarray(models["1"]), model_1), \
        "incorrect models['1'] values"
    assert np.array_equal(np.asarray(models["2"]), model_2), \
        "incorrect models['2'] values"


def test_evaluate():

    # pull result from SAMBA to check against
    results_file = cwd / 'samba_results.txt'
    samba_arrays = np.loadtxt(str(results_file), delimiter=',')

    # split up into arrays for each test
    samba_loworder = samba_arrays[0]
    samba_highorder = samba_arrays[1]
    samba_lowstd = samba_arrays[2]
    samba_highstd = samba_arrays[3]

    # check array equality within a tolerance
    predict = []
    for i in models.keys():
        predict.append(models[i].evaluate(g))

    # assert equality within a tolerance for means
    assert np.allclose(samba_loworder, np.asarray(predict[0][0])), \
        "incorrect evaluation for small-g"
    assert np.allclose(samba_highorder, np.asarray(predict[1][0])), \
        "incorrect evaluation for large-g"

    # assert equality within a tolerance for standard deviations
    assert np.allclose(np.sqrt(samba_lowstd), np.asarray(predict[0][1])), \
        "incorrect evaluation for small-g"
    assert np.allclose(np.sqrt(samba_highstd), np.asarray(predict[1][1])), \
        "incorrect evaluation for large-g"


def test_init():

    # check passing of variables into Multivariate class
    assert mixed.model_dict == models, "class variable self.model_dict not set"
    assert mixed.n_models == n_models, "class variable self.n_models not set"
    assert np.array_equal(mixed.x, g), "class variable self.x not set"


def test_mixing():

    # call samba_results.txt file
    results_file = cwd / 'samba_results.txt'
    samba_arrays = np.loadtxt(results_file, delimiter=',')

    # split up for these tests
    samba_mean = samba_arrays[4]
    samba_intervallow = samba_arrays[5]
    samba_intervalhigh = samba_arrays[6]
    samba_std = samba_arrays[7]

    # check predict function variables
    assert mixed.ci == ci, "ci is not passing"
    assert mixed.prediction is not None, "prediction is None"
    assert mixed.var_weights is not None, "var_weights is None"

    # check posterior draws
    assert posterior_draws == 0.0, "posterior draws is not zero"

    # check specific values of the mixed result
    assert np.allclose(samba_mean, mixed_mean), \
        "mixed mean not matching"
    assert np.allclose(samba_intervallow, mixed_intervals[0]), \
        "lower interval not matching"
    assert np.allclose(samba_intervalhigh, mixed_intervals[1]), \
        "higher interval not matching"
    assert np.allclose(samba_std, std_dev), "standard deviation not matching"


def test_evaluate_weights():

    # call samba_results.txt file
    results_file = cwd / 'samba_results.txt'
    samba_arrays = np.loadtxt(results_file, delimiter=',')

    # separate out weights
    weights_low = samba_arrays[8]
    weights_high = samba_arrays[9]

    # test weights
    assert np.allclose(weights_low, mixed.var_weights[0, :]), \
        "weights are incorrect"
    assert np.allclose(weights_high, mixed.var_weights[1, :]), \
        "weights are incorrect"

    # now that predict has been run, test pulling the weights
    assert np.array_equal(weights, mixed.var_weights), \
        "weights are not matching"

