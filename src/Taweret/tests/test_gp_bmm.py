###########################################################
# Testing file for gp_bmm.py (GP-based model mixing)
# This test suite covers the code written in the two
# tutorial notebooks for this mixing module.
# Author: Alexandra C. Semposki
# Date: 19 May 2026
###########################################################

# import sys
from pathlib import Path
from Taweret.mix.gp_bmm import GPmixing, GPRwrapper
from Taweret.models.samba_models import Loworder, Highorder
from Taweret.utils.kernels import SigmoidChangepoint
import numpy as np
from scipy.linalg import block_diag
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Set Taweret Path
dirname = Path(__file__).absolute()
cwd = dirname.parent

# now we write out the models we need
orders = [3, 3]
model_1 = Loworder(order=orders[0])
model_2 = Highorder(order=orders[1])

# make the model dict
models = {
    "1": model_1,
    "2": model_2
}

# evaluation of the models
g = np.linspace(1e-6, 1.0, 100)
predict = []

for i in models.keys():
    predict.append(models[i].evaluate(g))


# make covariance from models for testing
def build_cov_from_errors(x, sigma, ell=10.0, nugget=1e-6):
    x = np.asarray(x)
    sigma = np.asarray(sigma)

    dx = x[:, None] - x[None, :]
    rho = np.exp(-0.5 * (dx / ell)**2)

    cov = np.outer(sigma, sigma) * rho
    cov += nugget * np.eye(len(x))
    return cov


# make some training data
x1 = g[:30:3]
x2 = np.append(g[40:-1:5], g[-1])

# cut the data as well
data1 = predict[0][0][:30:3]
sigma1 = predict[0][1][:30:3]
data2 = np.append(predict[1][0][40:-1:5], predict[1][0][-1])
sigma2 = np.append(predict[1][1][40:-1:5], predict[1][1][-1])

# set up covariance functions here
cov1 = build_cov_from_errors(x1, sigma1, ell=0.1, nugget=1e-6)
cov2 = build_cov_from_errors(x2, sigma2, ell=1.0, nugget=1e-6)

# concatenate the training data
x_train = np.concatenate((x1, x2)).reshape(-1, 1)
y_train = np.concatenate((data1, data2)).reshape(-1, 1)

# covariance matrix
alpha = block_diag(cov1, cov2)

# set up the GP prior kernel
kernel = (C(constant_value=1.1, constant_value_bounds=[0.25, 2.25])
          * RBF(length_scale=0.5, length_scale_bounds=[0.02, 0.1]))

# hyperprior parameters
prior_params = {
                    'sigma': {'mu': 1.0, 'sig': 0.25},
                    'lengthscale': {'mu': 0.1, 'sig': 0.05}
                }

# set up the GP mixing class
gpmix = GPmixing(g, models, alpha=alpha, kernel=kernel, priors=True,
                 prior_params=prior_params, prior_choice='rbfnorm')

# run the unconstrained prior
_, draws = gpmix.prior_predict(sample=True, n_samples=10)

# run the evaluation on this prior
evalprior = gpmix.evaluate(x_train)

# train the kernel hyperparameters
gpmix.train(x_train, y_train)

# predict using this trained kernel
predictions = gpmix.predict()

# print out MAP values
map_vals = gpmix.map

# now we create nonstationary case (Model 1)
k1_ls = 0.15
k1_c2 = 500.0
k1 = (C(constant_value=k1_c2, constant_value_bounds='fixed')
      * RBF(length_scale=k1_ls, length_scale_bounds='fixed'))

# build the object
gp1 = GPRwrapper(
    kernel=k1,
    alpha=cov1,
)

# fit the nonstationary kernel
gp1.fit(x1.reshape(-1, 1), data1.reshape(-1, 1))

# predictions with kernel
gp1_predict, gp1_std_predict = gp1.predict(g.reshape(-1, 1), return_std=True)
_, gp1_cov_predict = gp1.predict(g.reshape(-1, 1), return_cov=True)

# create second nonstationary kernel (Model 2)
k2_ls = 0.15
k2_c2 = 5.0
k2 = (C(constant_value=k2_c2, constant_value_bounds='fixed')
      * RBF(length_scale=k2_ls, length_scale_bounds='fixed'))

gp2 = GPRwrapper(
    kernel=k2,
    alpha=cov2,
)

gp2.fit(x2.reshape(-1, 1), data2.reshape(-1, 1))

gp2_predict, gp2_std_predict = gp2.predict(g.reshape(-1, 1), return_std=True)
_, gp2_cov_predict = gp2.predict(g.reshape(-1, 1), return_cov=True)

# construct changepoint kernel as test
kernelCP = SigmoidChangepoint(ls1=k1_ls, ls2=k2_ls, cbar1=k1_c2, cbar2=k2_c2,
                              changepoint=0.3,
                              changepoint_bounds=[0.2, 0.6], width=0.2,
                              width_bounds=[0.1, 0.3])

# set up prior dict for the different hyperparameters
prior_dict = {
    'w': 'truncnorm',
    'cp': 'truncnorm',
    'switch': 'sigmoid',
}

# set up the model mixing step
gpmix2 = GPmixing(x=g, models=models, alpha=alpha, kernel=kernelCP,
                  mean_function="zero", priors=True, prior_params=None,
                  prior_choice='changepoint', prior_type=prior_dict,
                  switch='sigmoid', nopt=5000)

# fit and predict using the changepoint kernel
gpmix2.train(x_train.reshape(-1, 1), y_train.reshape(-1, 1),
             prior_choice='changepoint', prior_type=prior_dict,
             switch='sigmoid')

# predictions
gpmix_pred = gpmix2.predict()

###########################################################
# testing suite
###########################################################


def test_models():

    # test individual models
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


def test_kernel():

    # check kernel
    assert kernel is not None, "kernel is not set"


def test_init():

    # check variable passing
    assert gpmix.model_dict == models, "class variable self.model_dict not set"
    assert gpmix.prior_params == prior_params, '''class variable
    self.prior_params not set'''
    assert np.array_equal(gpmix.alpha, alpha), "class variable alpha not set"
    assert np.array_equal(gpmix.x, g), "class variable self.x not set"
    assert gpmix.kernel == kernel, "class variable kernel not set"


def test_draws():

    # number of draws
    assert draws is not None, "set number of draws"
    assert len(draws.T) == 10, "check number of draws"


def test_stationary():

    # test predictions
    assert predictions is not None, "check training step"
    assert map_vals is not None, "MAP values are not being calculated"


def test_nonstationary():

    # test kernels and results
    assert k1 is not None, "first kernel not set"
    assert k2 is not None, "second kernel not set"
    assert gp1 is not None, "first GP object is not set"
    assert gp2 is not None, "second GP object is not set"
    assert np.array_equal(gpmix2.x, g), "class variable self.x not set"
    assert kernelCP is not None, "changepoint kernel not set"
    assert gpmix_pred is not None, "GP predictions not calculated"
