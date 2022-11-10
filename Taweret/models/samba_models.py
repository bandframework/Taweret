# This toy example is copied from Alexandra; SAMBA package
# : \phi^4 expansion for small and large g :

import numpy as np
import sys

from Taweret.core.base_model import BaseModel

sys.path.append("../SAMBA/")

try:
    from SAMBA.samba import models   # assuming you have SAMBA in your Taweret top directory 
    from SAMBA.samba import mixing

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

    def log_likelihood_elementwise(self, model : object, x_exp : np.ndarray, y_exp : np.ndarray, y_err : np.ndarray, model_param=np.array([])) -> np.ndarray:
        '''
        Predict the log normal log likelihood for each experimental data point
        in the small-g model.

        Parameters
        ---------
        model : object
            model object with a predict method
        x_exp : np.1darray
            input parameter values for experimental data
        y_exp : np.1darray
            mean of the experimental data
        y_err : np.1darray
            standard deviation of the experimental data

        Returns:
        --------
        diff : np.1darray 
            The log likelihood for the array of 
            experimental points
        '''
        if model_param.size == 0:
            predictions, model_err = model.predict(x_exp)
        else:
            predictions, model_err = model.predict(x_exp, model_param)
        sigma = np.sqrt(np.square(y_err) + np.square(model_err))
        diff = -0.5* np.square((predictions.flatten() - y_exp)/ sigma) \
            - 0.5 * np.log(2*np.pi)- np.log(sigma)
        return diff
        

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

    def log_likelihood_elementwise(self, model : object, x_exp : np.ndarray, y_exp : np.ndarray, y_err : np.ndarray, model_param=np.array([])) -> np.ndarray:
        '''
        Predict the log normal log likelihood for each experimental data point
        in the large-g model.

        Parameters
        ---------
        model : object
            model object with a predict method
        x_exp : np.1darray
            input parameter values for experimental data
        y_exp : np.1darray
            mean of the experimental data
        y_err : np.1darray
            standard deviation of the experimental data

        Returns:
        --------
        diff : np.1darray 
            The log likelihood for the array of 
            experimental points
        '''
        if model_param.size == 0:
            predictions, model_err = model.predict(x_exp)
        else:
            predictions, model_err = model.predict(x_exp, model_param)
        sigma = np.sqrt(np.square(y_err) + np.square(model_err))
        diff = -0.5* np.square((predictions.flatten() - y_exp)/ sigma) \
            - 0.5 * np.log(2*np.pi)- np.log(sigma)
        return diff

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

    def log_likelihood_elementwise(self):
        '''
        Return the log likelihood for an array of 
        experimental points. 
        Not needed for this model.
        '''
        return None 

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
        '''
        Returns the log likelihood of an array of 
        experimental points. 
        Not needed for this model. 
        '''
        return None

    def set_prior(self):
        '''
        Set the prior on any model parameters.
        Not needed for this model. 
        '''
        return None 