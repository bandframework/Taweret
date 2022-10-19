# This toy example is copied from Alexandra; SAMBA package
# : \pi^4 expansion for small and large g :

import numpy as np
import math
from scipy import special, integrate
import sys

sys.path.append("../../../../SAMBA/")

try:
    from samba import models
    from samba import mixing
except Exception as e:
    print(e)
    print('To use the SAMBA toy models, SAMBA package needed to be installed first. cloning the SMABA github repo to the same place where your local Taweret github repo exist will also work.')

class loworder():
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

    def predict(self, input_values : np.array) -> np.array:
        """
        Predict the mean and error for given input values

        Parameters
        ----------
        input_values : numpy 1darray
            coupling strength (g) values
        """

        order = self.order
        M = models.Models(order, order)
        mean = M.low_g(input_values)
        U = models.Uncertainties(self.error_model)
        var = U.variance_low(input_values, order)

        return mean, np.sqrt(var)


class highorder():
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

    def predict(self, input_values : np.array) -> np.array:
        """
        Predict the mean and error for given input values

        Parameters
        ----------
        input_values : numpy 1darray
            coupling strength (g) values
        """

        order = self.order
        M = models.Models(order, order)
        mean = M.high_g(input_values)
        U = models.Uncertainties(self.error_model)
        var = U.variance_high(input_values, order)

        return mean, np.sqrt(var)
    
class true_model():
    """
    A wrapper for SAMBA  true function
    """

    def predict(self, input_values : np.array) -> np.array:
        """
        Predict the mean and error for given input values
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

class exp_data():
    """
    A wrapper for SAMBA  true function

    """

    def predict(self, input_values : np.array, error = 0.01) -> np.array:
        """
        Predict the mean and error for given input values

        Parameters
        ----------
        input_values : numpy 1darray
            coupling strength (g) values
        error : float
            defines the relative error as a fraction between (0,1)
        """

        order = 1
        M = mixing.LMM(order, order, error_model='informative')
        mean, sigma = M.add_data( input_values, input_values, error=error, plot=False)
        return mean, sigma



