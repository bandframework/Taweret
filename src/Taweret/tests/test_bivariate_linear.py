########################################
# Testing file for bivariate_linear.py #
# Author: Dananjaya Liyanage           #
# Date: 08/11/2023                     #
########################################

# Functions in this file calculate mixed
# log likelihood for three mixing methods.
# Then the same mix log likelihood is calculated
# by using Taweret package.
# Then the values are compared to make sure they are the same.
# The test can be upgraded to choose a random value for
# the two models to be evaluated and a random mixing function.

import os
import sys

# Set Taweret path
dirname = os.popen("find $PWD -type f -name test_bivariate_linear.py").read()
taweret_wd = dirname.split("test")[0]
sys.path.append(taweret_wd)

from pytest import approx
import pytest
import numpy as np
import bilby
from pytest import approx
from Taweret.models import coleman_models as toy_models
from Taweret.mix.bivariate_linear import BivariateLinear as BL


# sys.path.append('../Taweret')
# sys.path.append("../../Taweret")

# import sys
# sys.path.append("/Users/dananjayaliyanage/temp/Taweret")
# sys.path.append('../Taweret')


# Import coleman models

m1 = toy_models.coleman_model_1()
m2 = toy_models.coleman_model_2()
truth = toy_models.coleman_truth()

g = np.linspace(0, 9, 10)
plot_g = np.linspace(0, 9, 100)
true_output = truth.evaluate(plot_g)
exp_data = truth.evaluate(g)
yexp = np.array(exp_data[0]).reshape(-1, 1)
yexp_er = np.array(exp_data[1]).reshape(-1, 1)
# Make the following a random number to make this
# test more advanced

value = np.array([3])

mean1, var1, _ = m1.evaluate(g, value, full_corr=False)
mean2, var2, _ = m2.evaluate(g, value, full_corr=False)

# Mixing method

models = {'model1': m1, 'model2': m2}

mix_model_BMMC_mix = BL(
    models_dic=models,
    method='addstepasym',
    nargs_model_dic={
        'model1': 1,
        'model2': 1},
    same_parameters=False)

mix_model_BMMcor_mix = BL(
    models_dic=models,
    method='addstepasym',
    nargs_model_dic={
        'model1': 1,
        'model2': 1},
    same_parameters=False,
    BMMcor=True)

mix_model_mean_mix = BL(
    models_dic=models,
    method='addstepasym',
    nargs_model_dic={
        'model1': 1,
        'model2': 1},
    same_parameters=False,
    mean_mix=True)

mix_models = [mix_model_BMMC_mix, mix_model_BMMcor_mix, mix_model_mean_mix]

priors = bilby.core.prior.PriorDict()
priors['addstepasym_0'] = bilby.core.prior.Uniform(0, 9, name="addstepasym_0")
priors['addstepasym_1'] = bilby.core.prior.Uniform(0, 9, name="addstepasym_1")
priors['addstepasym_2'] = bilby.core.prior.Uniform(0, 1, name="addstepasym_2")
for mix_model in mix_models:
    mix_model.set_prior(priors)


def gaussian_LL(delta, Cov):
    inv_Cov = np.linalg.inv(Cov)
    det = np.linalg.det(Cov)
    inside_exp = -0.5 * delta.T @ inv_Cov @ delta
    inside_exp = inside_exp.flatten()
    # norm = np.sqrt(((2*np.pi)**len(delta))*det)
    n = len(delta)
    norm_const = -n / (2. * np.log(2. * np.pi))

    return inside_exp - 0.5 * np.log(det) + norm_const


def gaussian_L_elements(delta, Cov):

    sigma = np.sqrt(np.diag(Cov))
    diff = -0.5 * np.square((delta) / sigma) \
        - 0.5 * np.log(2 * np.pi) - np.log(sigma)
    return diff


def L_BMMoC(delta_y1, delta_y2, Cov, w):
    L1 = np.exp(gaussian_L_elements(delta_y1, Cov))
    L2 = np.exp(gaussian_L_elements(delta_y2, Cov))
    L = L1 * w + L2 * (1 - w)
    L = np.log(L)
    return np.sum(L)


def W_matrices(w):
    W1 = np.diag(w)
    W2 = np.diag(1 - w)
    return W1, W2


def L_BMMcor(delta_y1, delta_y2, Cov, w):
    W_1, W_2 = W_matrices(w)
    delta1 = W_1 @ delta_y1
    delta2 = W_2 @ delta_y2

    L1 = gaussian_LL(delta1, Cov)
    L2 = gaussian_LL(delta2, Cov)
    L = L1 + L2
    return L


def L_BMMmean(delta_y1, delta_y2, Cov, w):
   # W_1, W_2 = W_matrices(w)
    delta1 = w * delta_y1
    delta2 = (1 - w) * delta_y2

    delta = delta1 + delta2

    L = gaussian_LL(delta, Cov)
    # L2 = gaussian_L(delta2, Cov)
    # L = L1*L2
    return L[0]


# parameters for mixing function
# we can make these random numbers
# to make this test more advanced.

mix_param = np.array([3, 6, 0.5])

delta_y1 = mean1 - exp_data[0]
delta_y2 = mean2 - exp_data[0]
Cov = np.diag(np.square(exp_data[1]))
w1, w2 = mix_models[0].evaluate_weights(mix_param, g)

methods_name = ['BMMC', 'BMMcor', 'BMMmean']

test_mixing_func = [L_BMMoC, L_BMMcor, L_BMMmean]


def test_BMMC():
    i = 0
    model = mix_models[i]
    print(f'Testing {methods_name[i]} from bivariate_linear.py')
    log_lik_from_taweret_model = model.mix_loglikelihood(mix_param,
                                                         [value, value],
                                                         g,
                                                         yexp,
                                                         yexp_er)
    log_like_from_test = test_mixing_func[i](delta_y1, delta_y2, Cov, w1)
    # print(type(log_lik_from_taweret_model))
    # print(log_lik_from_taweret_model)

    # print(type(log_like_from_test))
    # print(log_like_from_test)

    log_like_from_test = approx(log_lik_from_taweret_model)


def test_BMMcor():
    i = 1
    model = mix_models[i]
    print(f'Testing {methods_name[i]} from bivariate_linear.py')
    log_lik_from_taweret_model = model.mix_loglikelihood(mix_param,
                                                         [value, value],
                                                         g,
                                                         yexp,
                                                         yexp_er)
    log_like_from_test = test_mixing_func[i](delta_y1, delta_y2, Cov, w1)
    # print(type(log_lik_from_taweret_model))
    # print(log_lik_from_taweret_model)

    # print(type(log_like_from_test))
    # print(log_like_from_test)

    log_like_from_test = approx(log_lik_from_taweret_model)


def test_BMMmean():
    i = 2
    model = mix_models[i]
    print(f'Testing {methods_name[i]} from bivariate_linear.py')
    log_lik_from_taweret_model = model.mix_loglikelihood(mix_param,
                                                         [value, value],
                                                         g,
                                                         yexp,
                                                         yexp_er)
    log_like_from_test = test_mixing_func[i](delta_y1, delta_y2, Cov, w1)
    # print(type(log_lik_from_taweret_model))
    # print(log_lik_from_taweret_model)

    # print(type(log_like_from_test))
    # print(log_like_from_test)

    log_like_from_test = approx(log_lik_from_taweret_model)

# def test_three_mixing_methods():
#     for i, model in enumerate(mix_models):
#         print(f'Testing {methods_name[i]} from bivariate_linear.py')
#         log_lik_from_taweret_model = model.mix_loglikelihood(mix_param,
#                                                              [value, value],
#                                                              g,
#                                                              yexp,
#                                                              yexp_er)
#         log_like_from_test = test_mixing_func[i](delta_y1, delta_y2, Cov, w1)
#         print(type(log_lik_from_taweret_model))
#         print(log_lik_from_taweret_model)

#         print(type(log_like_from_test))
#         print(log_like_from_test)

#         log_like_from_test = approx(log_lik_from_taweret_model)
#         # assert np.allclose(log_lik_from_taweret_model, log_like_from_test,
#         #                    "log likelihood calculated in test are different\
#         #                      from taweret bivariate linear methods")

