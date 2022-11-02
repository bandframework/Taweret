from logging import raiseExceptions
import numpy as np
from scipy.special import expit
from scipy.stats import norm, beta, dirichlet
#define log likelihood to be calculated give a model with a predict function
# and experimental measurments. 

eps = 1e-15
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
        predictions, model_err = model.predict(x_exp, model_param)
    sigma = np.sqrt(np.square(y_err) + np.square(model_err))
    diff = -0.5* np.square((predictions.flatten() - y_exp)/ sigma) \
        - 0.5 * np.log(2*np.pi)- np.log(sigma)
    return diff

def likelihood_elementwise(model : object, x_exp : np.ndarray, y_exp : np.ndarray, y_err : np.ndarray, model_param=np.array([])) -> np.ndarray:
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
        predictions, model_err = model.predict(x_exp)
    else:
        predictions, model_err = model.predict(x_exp, model_param)
    sigma = np.sqrt(np.square(y_err) + np.square(model_err))
    pre_factor = np.sqrt(2.0 * np.pi) * sigma
    diff = -0.5* np.square((predictions.flatten() - y_exp)/ sigma) 
    return np.exp(diff)/pre_factor


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
        return w, 1 - w
    elif method=='step':
        x_0 = mixture_params[0]
        w = np.array([1-(eps) if xi<=x_0 else eps for xi in x]).flatten()
        return w, 1 - w
    elif method=='cdf':
        theta_0, theta_1 = mixture_params
        x = theta_0 + theta_1*x
        w = norm.cdf(x)
        return w, 1 - w
    elif method=='beta':
        print('Warning: mixture_function - the `beta` choice forces a stochastic')
        print('         likelihood to be returned after calibration')
        w = beta.rvs(*mixture_params)
        return w, 1 - w
    elif method=='dirchlet':
        print('Warning: mixture_function - the `dirichlet` choice forces a stochastic')
        print('         likelihood to be returned after calibration')
        w = dirichlet.rvs(mixture_params)
        return w
    elif method=='switchcos':
        g1, g2, g3 = mixture_params
        w = np.array(list(map(lambda x: switchcos(g1, g2, g3, x), x)))
    return w, 1-w
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
        #return -np.inf
        # let's throw an error instead.
        raise Exception(f'g1 > g3 > g2 but given g1:{g1} g2:{g2} g3:{g3}')

    if x <= g1:
        return 1.0
    
    elif x <= g3:
        return (1.0 + np.cos((np.pi/2.0) * ((x - g1)/(g3 - g1))))/2.0
    
    elif x < g2:
        return 0.5 + np.cos((np.pi/2.0) * (1.0 + ((x - g3)/(g2 - g3))))/2.0
    
    else:
        return 0.0
