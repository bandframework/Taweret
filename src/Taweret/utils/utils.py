from scipy.linalg import lapack
from logging import raiseExceptions
import numpy as np
from scipy.special import expit
from scipy.stats import norm, beta, dirichlet
# define log likelihood to be calculated give a model with a predict function
# and experimental measurments.
# quick fix untill I find a permentant solution to put normed likelihood
# calculation code here

# Path to Jetscape model source code
import sys
sys.path.append(
    "/Users/dananjayaliyanage/git/Taweret/subpackages/js-sims-bayes/src")
# Imports from Jetscape code. Need to load the saved emulators.
# from configurations import *
# from emulator import *
# from bayes_mcmc import normed_mvn_loglike

eps = 1e-15


def normed_mvn_loglike(y, cov):
    """
    Evaluate the multivariate-normal log-likelihood for difference vector `y`
    and covariance matrix `cov`:

        log_p = -1/2*[(y^T).(C^-1).y + log(det(C))] + const.

    This likelihood IS NORMALIZED.
    The normalization const = -n/2*log(2*pi), where n is the dimensionality.

    Arguments `y` and `cov` MUST be np.arrays with dtype == float64 and shapes
    (n) and (n, n), respectively.  These requirements are NOT CHECKED.

    The calculation follows algorithm 2.1 in Rasmussen and Williams (Gaussian
    Processes for Machine Learning).

    """

    # Compute the Cholesky decomposition of the covariance.
    # Use bare LAPACK function to avoid scipy.linalg wrapper overhead.
    L, info = lapack.dpotrf(cov, clean=False)

    if info < 0:
        raise ValueError(
            'lapack dpotrf error: '
            'the {}-th argument had an illegal value'.format(-info)
        )
    elif info < 0:
        raise np.linalg.LinAlgError(
            'lapack dpotrf error: '
            'the leading minor of order {} is not positive definite'
            .format(info)
        )

    # Solve for alpha = cov^-1.y using the Cholesky decomp.
    alpha, info = lapack.dpotrs(L, y)

    if info != 0:
        raise ValueError(
            'lapack dpotrs error: '
            'the {}-th argument had an illegal value'.format(-info)
        )

    n = len(y)
    norm_const = -n / (2. * np.log(2. * np.pi))
    # print(norm_const)
    # print(L.diagonal())
    # return -.5*np.dot(y, alpha) - np.log(eps+L.diagonal()).sum() + norm_const
    return -.5 * np.dot(y, alpha) - np.log(L.diagonal()).sum() + norm_const


def normal_log_likelihood_elementwise(
    model: object,
    x_exp: np.ndarray,
    y_exp: np.ndarray,
    y_err: np.ndarray,
    model_param=np.array(
        [])) -> np.ndarray:
    """
    predict the log normal log liklihood for each experimental data point

    Parametrs
    ---------
    model : object
        model object with a predict method
    x_exp : np.1darray
        input parameter values for experimental data
    y_exp : np.1darray
        mean of the experimental data
    y_err : np.1darray
        standard deviation of the experimental data
    """
    if model_param.size == 0:
        try:
            predictions, model_err = model.evaluate(x_exp)
        except BaseException:
            predictions, model_err, _ = model.evaluate(x_exp)

    else:
        try:
            predictions, model_err = model.evaluate(x_exp, model_param)
        except BaseException:
            predictions, model_err, _ = model.evaluate(x_exp, model_param)

    sigma = np.sqrt(np.square(y_err) + np.square(model_err))
    diff = -0.5 * np.square((predictions.flatten() - y_exp) / sigma) \
        - 0.5 * np.log(2 * np.pi) - np.log(sigma)
    return diff


def normal_likelihood_elementwise(
    model: object,
    x_exp: np.ndarray,
    y_exp: np.ndarray,
    y_err: np.ndarray,
    model_param=np.array(
        [])) -> np.ndarray:
    """
    predict the normal liklihood for each experimental data point

    Parametrs
    ---------
    model : object
        model object with a predict method
    x_exp : np.1darray
        input parameter values for experimental data
    y_exp : np.1darray
        mean of the experimental data
    y_err : np.1darray
        standard deviation of the experimental data
    """
    if model_param.size == 0:
        try:
            predictions, model_err = model.evaluate(x_exp)
        except BaseException:
            predictions, model_err, _ = model.evaluate(x_exp)

    else:
        try:
            predictions, model_err = model.evaluate(x_exp, model_param)
        except BaseException:
            predictions, model_err, _ = model.evaluate(x_exp, model_param)

    sigma = np.sqrt(np.square(y_err) + np.square(model_err))
    pre_factor = np.sqrt(2.0 * np.pi) * sigma
    diff = -0.5 * np.square((predictions.flatten() - y_exp) / sigma)
    return np.exp(diff) / pre_factor


def mixture_function(
        method: str,
        x: np.ndarray,
        mixture_params: np.ndarray,
        prior=None) -> np.ndarray:
    """
    predict the weights from the mixture funtion at the give input parameter values x

    Parameters
    ----------
    method : str
        name of the linear mixing function method
    x : np.1darray
        input parameter values
    mixture_params : np.1darray
        parametrs that decide the shape of mixture function
    prior : (optional) bilby prior object
        Used only in step mixing
        to deal with negative values of the input.
    """

#     if mixture_params.size == 0:
#         return np.ones(len(x))
    if method == 'sigmoid':
        theta_0, theta_1 = mixture_params
        w = expit((x - theta_0) / theta_1)
        return w, 1 - w
    elif method == 'step':

        x_0 = mixture_params[0]
        if x_0 >= 0:
            # If x is less than x_0 it's 1. Otherwise 0.
            w = np.array(
                [1 - (eps) if xi <= x_0 else eps for xi in x]).flatten()
        elif x_0 < 0:
            # x_0 = -1*x_0
            max_num = prior['step_0'].maximum
            bound = max_num + x_0
            # If x is greater than max_num - |x_0| it's 1. Otherwise 0.
            w = np.array(
                [1 - (eps) if xi >= bound else eps for xi in x]).flatten()
        return w, 1 - w
    elif method == 'addstep':

        x_0, alpha = mixture_params
        if x_0 >= 0:
            # If x is less than x_0 it's 1. Otherwise 0.
            w1 = np.array(
                [1 - (eps) if xi <= x_0 else eps for xi in x]).flatten()
            max_num = prior['addstep_0'].maximum
            bound = max_num - x_0
            # If x is greater than max_num - |x_0| it's 1. Otherwise 0.
            w2 = np.array(
                [1 - (eps) if xi >= bound else eps for xi in x]).flatten()
        else:
            raise Exception(f'x_0 has to be non negative but provided {x_0}')
        w = alpha * w1 + (1 - alpha) * w2
        return w, 1 - w

    elif method == 'addstepasym':

        x_0, x_1, alpha = mixture_params
        if x_0 >= 0:
            # If x is less than x_0 it's 1. Otherwise 0.
            w1 = np.array(
                [1 - (eps) if xi <= x_0 else eps for xi in x]).flatten()
            # max_num = prior['addstepasym_0'].maximum
            # bound = max_num - x_0
            # If x is greater than max_num - |x_0| it's 1. Otherwise 0.
            # w2 = np.array([1-(eps) if xi>=bound else eps for xi in x]).flatten()
        else:
            raise Exception(f'x_0 has to be non negative but provided {x_0}')

        if x_1 >= 0:
            # If x is less than x_0 it's 1. Otherwise 0.
            w2 = np.array(
                [1 - (eps) if xi >= x_1 else eps for xi in x]).flatten()
            # max_num = prior['addstep_0'].maximum
            # bound = max_num - x_0
            # If x is greater than max_num - |x_0| it's 1. Otherwise 0.
            # w2 = np.array([1-(eps) if xi>=bound else eps for xi in x]).flatten()
        else:
            raise Exception(f'x_1 has to be non negative but provided {x_1}')

        w = alpha * w1 + (1 - alpha) * w2
        return w, 1 - w

    elif method == 'cdf':
        theta_0, theta_1 = mixture_params
        x = theta_0 + theta_1 * x
        w = norm.cdf(x)
        w = np.array(w).flatten()
        return w, 1 - w
    elif method == 'beta':
        print('Warning: mixture_function - the `beta` choice forces a stochastic')
        print('         likelihood to be returned after calibration')
        w = beta.rvs(*mixture_params)
        return w, 1 - w
    elif method == 'dirchlet':
        print('Warning: mixture_function - the `dirichlet` choice forces a stochastic')
        print('         likelihood to be returned after calibration')
        w = dirichlet.rvs(mixture_params)
        return w
    elif method == 'switchcos':
        g1, g2, g3 = mixture_params
        w = np.array(list(map(lambda x: switchcos(g1, g2, g3, x), x)))
        return w, 1 - w
    elif method == 'calibrate_model_1':

        w = np.ones(len(x))
        return w, 1 - w
    elif method == 'calibrate_model_2':

        w = np.ones(len(x))
        return 1 - w, w

    else:
        raise Exception('Method is not available for `mixture function`')


def switchcos(g1, g2, g3, x):
    """Switchcos function in Alexandras Samba module
    link https://github.com/asemposki/SAMBA/blob/0479b4deff46f3cb77b82bb30abd5693de8980f3/samba/mixing.py#L1205

    Parameters:
    -----------
    g1 : float
        switching value from left constant to first cosine
    g2 : float
        switching value from second cosine to right constant
    g3 : float
        switching value from first cosine to second cosine
    x : float
        the input parameter value to calculate the weight
    """

    if g1 > g2 or g2 < g3 or g1 > g3:
        # return -np.inf
        # let's throw an error instead.
        raise Exception(f'g1 > g3 > g2 but given g1:{g1} g2:{g2} g3:{g3}')

    if x <= g1:
        return 1.0

    elif x <= g3:
        return (1.0 + np.cos((np.pi / 2.0) * ((x - g1) / (g3 - g1)))) / 2.0

    elif x < g2:
        return 0.5 + np.cos((np.pi / 2.0) *
                            (1.0 + ((x - g3) / (g2 - g3)))) / 2.0

    else:
        return 0.0
