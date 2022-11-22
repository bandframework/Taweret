import numpy as np
import math
from scipy.special import expit
from scipy.stats import norm, beta, dirichlet

# define log likelihood to be calculated give a model with a predict function
# and experimental measurments.


def log_of_normal_dist(mean: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return -((mean - mu) ** 2) / (2 * sigma) ** 2 - 0.5 * np.log(
        2 * np.pi * sigma**2
    )


eps = 1e-20


def log_likelihood_elementwise(
    model: object,
    x_exp: np.ndarray,
    y_exp: np.ndarray,
    y_err: np.ndarray,
    model_param=np.array([]),
) -> np.ndarray:
    """
    predict the log normal log liklihood for each experimental data point

    Parametrs
    ---------
    model : object
        model object with a predict method
    x_exp : np.ndarray
        input parameter values for experimental data
    y_exp : np.ndarray
        mean of the experimental data
    y_err : np.ndarray
        standard deviation of the experimental data
    """
    if model_param.size == 0:
        predictions, model_err = model.predict(x_exp)
    else:
        predictions, model_err = model.predict(x_exp, model_param)
    sigma = np.sqrt(np.square(y_err) + np.square(model_err))
    diff = -0.5 * np.square(
        (predictions.flatten() - y_exp) / sigma
    ) - 0.5 * np.log(2 * math.pi * sigma * sigma)
    return diff


def mixture_function(
    method: str, x: np.ndarray, mixture_params: np.ndarray
) -> np.ndarray:
    """
    predict the weights from the mixture funtion at the give input parameter
    values x

    Parameters
    ----------
    method : str
        name of the linear mixing function method
    x : np.ndarray
        input parameter values
    mixture_params : np.ndarray
        parametrs that decide the shape of mixture function
    """

    if mixture_params.size == 0:
        return np.ones(len(x))
    if method == "sigmoid":
        theta_0, theta_1 = mixture_params
        w = expit((x - theta_0) / theta_1)
        return w, 1 - w
    elif method == "step":
        x_0 = mixture_params[0]
        w = np.array([1 - (eps) if xi <= x_0 else eps for xi in x]).flatten()
        return np.array([w, 1 - w])
    elif method == "cdf":
        theta_0, theta_1 = mixture_params
        x = theta_0 + theta_1 * x
        w = norm.cdf(x)
        return np.array([w, 1 - w])
    elif method == "beta":
        print(
            "Warning: mixture_function - the `beta` choice forces a stochastic"
        )
        print("         likelihood to be returned after calibration")
        w = beta.rvs(*mixture_params)
        return np.array([w, 1 - w])
    elif method == "dirchlet":
        print(
            "Warning: mixture_function - the `dirichlet` choice forces a stochastic"
        )
        print("         likelihood to be returned after calibration")
        w = dirichlet.rvs(mixture_params)
        return w
    else:
        raise Exception("Method is not available for `mixture_function`")
