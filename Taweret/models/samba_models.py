# This toy example is copied from Alexandra; SAMBA package
# : \pi^4 expansion for small and large g :

import numpy as np
import sys
sys.path.append("../../SAMBA/samba")

from Taweret.core.base_model import BaseModel

try:
    from Taweret.models.models_samba import Models, Uncertainties    # fix this later on to pull from samba
except Exception as e:
    print(e)
    print('To use the SAMBA toy models, SAMBA package needed to be installed first. \
        Cloning the SAMBA github repo to the same place where your local Taweret github repo exist will also work.')

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


    def evaluate(self, input_values : np.array) -> np.array:
        """
        Predict the mean and model error for given input values

        Parameters:
        ----------
        input_values : numpy 1darray
            coupling strength (g) values

        Returns:
        -------
        mean : numpy 1darray
            mean of the low order expansion
        
        np.sqrt(var) : numpy 1darray
            standard deviation of the low order expansion
        """

        order = self.order
  #      M = models.Models(order, order)
        M = Models(order, order)
        mean = M.low_g(input_values)
  #      U = models.Uncertainties(self.error_model)
        U = Uncertainties(self.error_model)
        var = U.variance_low(input_values, order)

        return mean, np.sqrt(var)


    def log_likelihood_elementwise(self):  # return Dan's log likelihood for the linear.py method
        '''
        
        '''
        return super().log_likelihood_elementwise()

    
    def set_prior(self):
        '''
        Set the prior for any parameters in the model. 
        Not needed in this model. 
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
            The model error for the chosen input 
            values
        """

        order = self.order
        M = Models(order, order)
        mean = M.high_g(input_values)
        U = Uncertainties(self.error_model)
        var = U.variance_high(input_values, order)

        return mean, np.sqrt(var)

    
    def log_likelihood_elementwise(self):   # return the log likelihood for Dan's linear method
        '''
        Calculate the log likelihood of the model.
        '''
        return super().log_likelihood_elementwise()

    
    def set_prior(self):
        '''
        Set the prior for any parameters in the model. 
        Not needed in this model. 
        '''
        return None

    
class true_model(BaseModel):
    """
    A wrapper for the true model in this 
    toy problem. 

    $$ \int dx \exp(-\frac{x^2}{2} - g^2 x^4) $$
    """

    def __init__(self):

        pass 

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
            The mean of the true model
        
        np.sqrt(var) : numpy 1darray
            The standard deviation. This returns an
            array of zeros
        """

        order = 1
        M = models.Models(order, order)
        mean = M.true_model(input_values)
        var = np.zeros(shape=mean.shape)
        return mean, np.sqrt(var)


    def log_likelihood_elementwise(self):
        '''
        Calculate the log likelihood of the model.
        Not needed for the true model.
        '''
        return None 

    def set_prior(self):
        '''
        Set the prior for the model.
        Not needed for the true model.
        '''
        return None