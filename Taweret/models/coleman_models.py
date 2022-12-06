import numpy as np
from Taweret.core.base_model import BaseModel
from Taweret.utils.utils import normal_log_likelihood_elementwise as log_likelihood_elementwise_utils
import bilby


class coleman_model_1(BaseModel):

    def __init__(self) -> None:
        self._prior = None

    def evaluate(self, input_values : np.array, model_param : np.array) -> np.array:
        """
        Predict the mean and error for given input values

        Parameters
        ----------
        input_values : numpy 1darray
            input parameter values
        model_param : numpy 1darray
            value of the model parameter
        """

        x = input_values.flatten()
        mean = np.zeros(len(x))
        var = 0.3*0.3*np.zeros(len(x))

        if len(model_param.flatten()) !=1 :
            raise TypeError('The model_param has to be single element numpy array')

        mean = 0.5 * (x + model_param.item()) - 2
        return mean, np.sqrt(var)

    def log_likelihood_elementwise(self, x_exp, y_exp, y_err, model_param):
        return log_likelihood_elementwise_utils(self, x_exp, y_exp, y_err, model_param)
    
    def set_prior(self, bilby_priors=None):
        '''
        Set the prior on model parameters.
        '''
        if bilby_priors is None:
            print('Using default priors for model 1')
            priors = bilby.prior.PriorDict()
            priors['model1_0']=bilby.core.prior.Uniform(1, 6, "model1_0")
        else:
            priors = bilby_priors
        print(priors)
        self._prior=priors
        return priors

    @property
    def prior(self):
        if self._prior is None:
            return self.set_prior()
        else:
            return self._prior

    @prior.setter
    def prior(self, bilby_prior_dic):
        return self.set_prior(bilby_prior_dic)

class coleman_model_2(BaseModel):

    def __init__(self) -> None:
        self._prior = None
        
    def evaluate(self, input_values : np.array, model_param : np.array) -> np.array:
        """
        Predict the mean and error for given input values

        Parameters
        ----------
        input_values : numpy 1darray
            input parameter values
        model_param : numpy 1darray
            value of the model parameter
        """

        x = input_values.flatten()
        mean = np.zeros(len(x))
        var = 0.3*0.3*np.zeros(len(x))

        if len(model_param.flatten()) !=1 :
            raise TypeError('The model_param has to be single element numpy array')

        mean = -0.5 * (x - model_param.item()) + 3.7
        return mean, np.sqrt(var)  

    def log_likelihood_elementwise(self, x_exp, y_exp, y_err, model_param):
        return log_likelihood_elementwise_utils(self, x_exp, y_exp, y_err, model_param)
    
    def set_prior(self, bilby_priors=None):
        '''
        Set the prior on model parameters.
        '''
        if bilby_priors is None:
            print('Using default priors for model 2')
            priors = bilby.prior.PriorDict()
            priors['model2_0']=bilby.core.prior.Uniform(-2, 3, "model2_0")
        else:
            priors = bilby_priors
        print(priors)
        self._prior=priors
        return priors

    @property
    def prior(self):
        if self._prior is None:
            return self.set_prior()
        else:
            return self._prior

    @prior.setter
    def prior(self, bilby_prior_dic):
        return self.set_prior(bilby_prior_dic)

class coleman_truth(BaseModel):

    def evaluate(self, input_values : np.array) -> np.array:
        """
        Predict the mean and error for given input values

        Parameters
        ----------
        input_values : numpy 1darray
            input parameter values
        """

        x = input_values.flatten()
        mean = np.zeros(len(x))
        var = 0.3* 0.3 * np.ones(len(x))

        mean = 2 - 0.1 * (x - 4)**2
        return mean, np.sqrt(var)

    def log_likelihood_elementwise(self, x_exp, y_exp, y_err, model_param):
        pass

    def set_prior(self, bilby_priors=None):
        pass