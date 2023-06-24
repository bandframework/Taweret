# This toy example is copied from Alexandra; SAMBA package
# : \phi^4 expansion for small and large g :

import numpy as np
import sys
sys.path.append('../core')

from Taweret.Taweret.core.base_model import BaseModel
from Taweret.Taweret.utils.utils import normal_log_likelihood_elementwise as log_likelihood_elementwise_utils

sys.path.append("../../../../SAMBA")

# from samba import models   # assuming you have SAMBA in your Taweret top directory 
# from samba import mixing

try:
    from SAMBA import models   # assuming you have SAMBA in your Taweret top directory 
    from SAMBA import mixing

except Exception as e:
    print(e)
    print('''To use the SAMBA toy models, SAMBA package needed to be installed first''')
    print('Then use `sys.path.append("path_to_local_SAMBA_installation")` in your code before calling \
SAMBA models')


class loworder(BaseModel):
    """
    A wrapper for SAMBA low order expansion function

    """

    def __init__(self, order, error_model='informative'):
        """
        Parameters
        ----------
        order : int
            Truncation order of expansion
        error_model : str
            Error calculation method. Either 'informative' or 'uninformative'
        
        Raises
        ------
        TypeError
            If the order is not an integer
        """

        if isinstance(order, int):
            self.order = order
        else:
            raise TypeError(f"order has to be an integer number: {order}")

        self.error_model = error_model
        self.prior = None

    def evaluate(self, input_values : np.array) -> np.array:
        """
        Evaluate the mean and error for given input values

        Parameters
        ----------
        input_values : numpy 1darray
            coupling strength (g) values
        
        Returns:
        --------
        mean : numpy 1darray
            The mean of the model
        np.sqrt(var) : numpy 1darray
            The truncation error of the model
        """

        order = self.order
        M = models.Models(order, order)
        mean = M.low_g(input_values)
        U = models.Uncertainties(self.error_model)
        var = U.variance_low(input_values, order)

        return mean, np.sqrt(var)

    def log_likelihood_elementwise(self, x_exp, y_exp, y_err, model_param):
        return log_likelihood_elementwise_utils(self, x_exp, y_exp, y_err, model_param)

    def set_prior(self):
        '''
        Set the prior on model parameters.
        Not needed for this model. 
        '''
        return None


class highorder(BaseModel):
    """
    A wrapper for SAMBA high order expansion function

    """   

    def __init__(self, order, error_model='informative'):
        """
        Parameters
        ----------
        order : int
            Truncation order of expansion
        error_model : str
            Error calculation method. Either 'informative' or 'uninformative'
        
        Raises
        ------
        TypeError
            If the order is not an integer
        """

        if isinstance(order, int):
            self.order = order
        else:
            raise TypeError(f"order has to be an integer number: {order}")

        self.error_model = error_model
        self.prior = None
        
    def evaluate(self, input_values : np.array) -> np.array:
        """
        Evaluate the mean and error for given input values

        Parameters
        ----------
        input_values : numpy 1darray
            coupling strength (g) values

        Returns:
        --------
        mean : numpy 1darray
            The mean of the model
        np.sqrt(var) : numpy 1darray
            The truncation error of the model
        """

        order = self.order
        M = models.Models(order, order)
        mean = M.high_g(input_values)
        U = models.Uncertainties(self.error_model)
        var = U.variance_high(input_values, order)

        return mean, np.sqrt(var)

    def log_likelihood_elementwise(self, x_exp, y_exp, y_err, model_param):
        return log_likelihood_elementwise_utils(self, x_exp, y_exp, y_err, model_param)

    def set_prior(self):
        '''
        Set the prior on the model parameters.
        Not needed for this model. 
        '''
        return None


class true_model(BaseModel):
    """
    A wrapper for SAMBA  true function
    """

    def evaluate(self, input_values : np.array) -> np.array:
        """
        Evaluate the mean and error for given input values
        Parameters
        ----------
        input_values : numpy 1darray
            coupling strength (g) values
        """

        order = 1
        M = models.Models(order, order)
        mean = M.true_model(input_values)
        var = np.zeros(shape=mean.shape)
        return mean, np.sqrt(var)

    def log_likelihood_elementwise(self, x_exp, y_exp, y_err):
        return log_likelihood_elementwise_utils(self, x_exp, y_exp, y_err)

    def set_prior(self):
        '''
        Set the prior on any model parameters.
        Not needed for this model. 
        '''
        return None 


class exp_data(BaseModel):    # --> check that this model is set up correctly
    """
    A wrapper for SAMBA data function

    """

    def evaluate(self, input_values : np.array, error = 0.01) -> np.array:
        """
        Evaluate the mean and error for given input values

        Parameters
        ----------
        input_values : numpy 1darray
            coupling strength (g) values
        error : float
            defines the relative error as a fraction between (0,1)
        """

        order = 1
        M = mixing.LMM(order, order, error_model='informative')
        mean, sigma = M.add_data(input_values, input_values, error=error, plot=False)
        
        return mean, sigma

    def log_likelihood_elementwise(self):
        return None

    def set_prior(self):
        '''
        Set the prior on any model parameters.
        Not needed for this model. 
        '''
        return None 
