from tkinter import W
import numpy as np
import math
from scipy.special import expit
from scipy.stats import norm
#define log likelihood to be calculated give a model with a predict function
# and experimental measurments. 

eps = 1e-20
def log_likelihood_elementwise(model : object, x_exp : np.ndarray, y_exp : np.ndarray, y_err : np.ndarray, model_param=np.array([])) -> np.ndarray:
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
        predictions, model_err = model.predict(x_exp)
    else:
        predictions, model_err = model.predict(model_param, x_exp)
    sigma = y_err + model_err
    diff = -0.5* np.square((predictions.flatten() - y_exp)/ sigma) \
        - 0.5 * np.log(2*math.pi*sigma*sigma)
    return diff

def mixture_function(method : str, x : np.ndarray, mixture_params : np.ndarray) -> np.ndarray:
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
    """
    
    if mixture_params.size == 0:
        return np.ones(len(x))
    if method=='sigmoid':
        theta_0, theta_1 = mixture_params
        w = expit((x-theta_0)/theta_1)
    elif method=='step':
        x_0 = mixture_params[0]
        w = np.array([1-(eps) if xi<=x_0 else eps for xi in x]).flatten()
    elif method=='cdf':
        theta_0, theta_1 = mixture_params
        x = theta_0 + theta_1*x
        w = norm.cdf(x)
    return w