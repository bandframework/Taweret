###########################################################
# Testing file for gaussian.py (SAMBA multivariate BMM)
# we can test both the samba_models and gaussian.py code 
# in this pytest script here
# Author: Alexandra Semposki
# Date: 03 April 2023
###########################################################

import pytest
import numpy as np

import sys
sys.path.append('../../Taweret')

from Taweret.models.samba_models import *
from Taweret.mix.gaussian import *
from Taweret.core.base_mixer import *
from Taweret.core.base_model import *

# set up the order to test at
order = 3

# List models to mix (form dict)
model_1 = loworder(order=order)
model_2 = highorder(order=order)

labels=[r'$N_s = 3$', r'$N_l = 3$']

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
    assert models is not None 
    assert np.array_equal(np.asarray(models["1"]), model_1), \
        "incorrect models['1'] values"
    assert np.array_equal(np.asarray(models["2"]), model_2), \
        "incorrect models['2'] values"

def test_evaluate():

    # pull result from SAMBA to check against
    samba_arrays = np.loadtxt('samba_results.txt', delimiter=',')

    # check array equality within a tolerance
    predict = []
    for i in models.keys():
        predict.append(models[i].evaluate(g))

    # assert equality within a tolerance for means
    assert np.allclose(samba_arrays[0], np.asarray(predict[0][0])), \
        "incorrect evaluation for small-g"
    assert np.allclose(samba_arrays[1], np.asarray(predict[1][0])), \
        "incorrect evaluation for large-g"
    
     # assert equality within a tolerance for standard deviations
    assert np.allclose(np.sqrt(samba_arrays[2]), np.asarray(predict[0][1])), \
        "incorrect evaluation for small-g"
    assert np.allclose(np.sqrt(samba_arrays[3]), np.asarray(predict[1][1])), \
        "incorrect evaluation for large-g"

def test_init():

    # check passing of variables into Multivariate class
    assert mixed.model_dict == models, "class variable self.model_dict not set"
    assert mixed.n_models == n_models, "class variable self.n_models not set"
    assert np.array_equal(mixed.x, g), "class variable self.x not set"

def test_mixing():

    # check predict function variables
    assert mixed.ci == ci, "ci is not passing"
    assert mixed.prediction is not None, "prediction is None"
    assert mixed.var_weights is not None, "var_weights is None"

    # check returned variables
    assert posterior_draws == 0.0, "posterior draws is not zero"
    assert mixed_mean is not None, "mixed_mean is None"
    assert mixed_intervals is not None, "mixed_intervals is None"
    assert std_dev is not None, "std_dev is None"

def test_evaluate_weights():

    # now that predict has been run, test the weights
    assert np.array_equal(weights, mixed.var_weights), "weights are incorrect"