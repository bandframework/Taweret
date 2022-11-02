# This will have all the linear bayesian model mixing methods.
# Takes Models as inputs:
# Check if Models have an predict method and they should output a mean and a variance.
#
# Modified by K. Ingles

from ..core.base_mixer import BaseMixer
from ..core.base_model import BaseModel
from ..utils.utils import log_of_normal_dist

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm, dirichlet
from typing import Any, Dict, List

eps = 1.0e-20


class LinearMixerGlobal(BaseMixer):
    """
    Generates a linear mixed model

    """

    def __init__(self, models, nargs_for_each_model={}, n_mix=0):
        """
        Parameters
        ----------
        models : Dict[str, Model(BaseModel)]
            models to mix, each must contain a evaluate method
        x_exp : np.1darray
            input parameter values of experimental data
        y_exp : np.1darray
            mean of the experimental data for each input value
        y_err : np.1darray
            standard deviation of the experimental data for each input value
        method : str
            mixing function
        nargs_for_each_model : Dict[str, int]
            number of free parameters for each model
        n_mix : int
            number of free parameters in the mixing funtion
        """

        # check that lengths of lists are compatible
        if len(models) != len(nargs_for_each_model) and len(nargs_for_each_model) != 0:
            raise Exception('in linear_mix.__init__: len(nargs_for_each_model) must either equal len(models) or 0')

        #check for predict method in the models
        for i, model in enumerate(models):
            try:
                issubclass(model, BaseModel)
            except AttributeError:
                print(f'model {i} needs to inherit from Taweret.core.base_model.BaseModel')
            else:
                continue

        self.models = models
        self.nargs_for_each_model = nargs_for_each_model
        self.n_mix = n_mix

        # function returns

    def evaluate(self, model_params: Dict[str, List[Any]]) -> np.ndarray:
        """
        evaluate mixed model for given mixing function parameters

        Parameters
        ----------
        mixture_params : np.1darray
            parameter values that fix the shape of mixing function
        model_params: Dict[str, List[Any]]
            list of parameter values for each model


        Returns
        -------
        prediction : np.float
            the prediction made for mixed model
        """

        weights = self.weights()

        if len(self.nargs_for_each_model) == 0:
            return np.sum([weight * model.evaluate()[0]
                           for weight, model in zip(weights, self.models.value())])
        else:
            return np.sum([weight * model.evaluate(*params)[0]
                           for weight, model, params in zip(weights, self.models.values(), model_params)])

    def evaluate_weights(self, mix_params) -> np.ndarray:
        """
        return the mixing function value at the input parameter values x

        Parameters
        ----------
        mixture_params : np.1darray
            parameter values that fix the shape of mixing function
        x : np.1darray
            input parameter values
        """
        return dirichlet(mix_params).rvs()

    def map(self):
        if self.has_trained:
            return self._map
        else:
            raise Exception("Please train model before requesting MAP")

    def mix_loglikelihood(self, y_exp, y_err, mix_params=[], model_params={}) -> float:
        """
        log likelihood of the mixed model given the mixing function parameters

        Parameters
        ----------
        mixture_params : np.1darray
            parameter values that fix the shape of mixing function
        model_params: list[np.1darray]
            list of model parameters for each model, note that different models can take different
            number of params
        """

        # In general, the weights statisfy a simplex condition (they sum to 1)
        # A natural distribution to choose for the weights is a dirichlet
        # distribution, which hyperparameters that need priors specified

        # return weights for different models and take logs
        weights = self.evaluate_weights(mix_params)
        log_weights = np.log(weights + eps)

        # calculate log likelihoods
        if len(self.nargs_for_each_model) == 0:
            log_likelis = np.array(
                [log_of_normal_dist(model.evaluate(), y_exp, y_err) + log_weight
                 for model, log_weight in zip(self.models, log_weights)])
        else:
            log_likelis = np.array(
                [log_of_normal_dist(model.evaluate(*params), y_exp, y_err) + log_weight
                 for model, params, log_weight in zip(self.models, model_params, log_weights)])

        total_sum = np.logaddexp.reduce(log_likelis)
        return total_sum.item()

    def plot_weights(self, mix_params: np.ndarray) -> np.ndarray:
        """
        plot the mixing function against the input parameter values x

        Parameters
        ----------
        mixture_params : np.1darray
            parameter values that fix the shape of mixing function
        x : np.1darray
            input parameter values
        """
        weights = self.evaluate_weights(mix_params)
        fig, ax = plt.subplots()
        ax.scatter(np.arange(len(weights)), weights)
        ax.set_xlabel('Model number')
        ax.set_ylabel('Model weight')
        return None

    def posterior(self):
        if self.has_trained:
            return self.posterior
        else:
            raise Exception("Please train model before requesting posterior")

    def predict(self):
        if self.has_trained:
            return None
        else:
            raise Exception("Please train model before making predictions")
        pass

    def prior(self):
        return self.prior

    def prior_predict(self):
        if self.has_trained:
            return None
        else:
            raise Exception("Please train model before making predictions")
    
    def sample_prior(self, num_samples):
        mix_params = [self.prior[name].rvs(num_sampels) for name in self.prior.keys()]
        return mix_params

    def set_prior(self):
        print("""Notice: For global fitting, it the prior is always assumed to
                 be a Dirichlet distribution""")
        scale = 1000
        prior_dict = {f'param_{1}': lognorm(scale) for i in range(self.n_mix)}
        self.prior = prior_dict


class LinearMixerLocal(BaseMixer):
    """
    Generates a linear mixed model

    """

    def __init__(self, models, x_exp, y_exp, y_err, method='sigmoid',
                 nargs_for_each_model=[], n_mix=0):
        """
        Parameters
        ----------
        models : Dict[str, Model(BaseModel)]
            models to mix, each must contain a evaluate method
        x_exp : np.1darray
            input parameter values of experimental data
        y_exp : np.1darray
            mean of the experimental data for each input value
        y_err : np.1darray
            standard deviation of the experimental data for each input value
        method : str
            mixing function
        nargs_for_each_model : list
            number of free parameters for each model
        n_mix : int
            number of free parameters in the mixing funtion
        """

        # check that lengths of lists are compatible
        if len(models) != len(nargs_for_each_model) and len(nargs_for_each_model) != 0:
            raise Exception('in linear_mix.__init__: len(nargs_for_each_model) must either equal len(models) or 0')

        #check for predict method in the models
        for i, model in enumerate(models):
            try:
                issubclass(model, BaseModel)
            except AttributeError:
                print(f'model {i} needs to inherit from Taweret.core.base_model.BaseModel')
            else:
                continue

        self.models = models

        #check if the dimensions match for experimental data
        if (x_exp.shape[0] != y_exp.shape[0]) or (x_exp.shape[0] != y_err.shape[0]):
            raise Exception('x_exp, y_exp, y_err all should have the same shape[0]')

        self.x_exp = x_exp.flatten()
        self.y_exp = y_exp.flatten()
        self.y_err = y_err.flatten()

        #check if mixing method exist
        if method not in ['step', 'sigmoid', 'cdf']:
            raise Exception('only supports the step or sigmoid mixing functions')

        self.method = method

        self.nargs_for_each_model = nargs_for_each_model
        self.n_mix = n_mix

        # function returns

    def mix_loglikelihood(self, mixture_params : np.ndarray, model_params=[]) -> float:
        """
        log likelihood of the mixed model given the mixing function parameters

        Parameters
        ----------
        mixture_params : np.1darray
            parameter values that fix the shape of mixing function
        model_params: list[np.1darray]
            list of model parameters for each model, note that different models can take different
            number of params
        """

        # In general, the weights statisfy a simplex condition (they sum to 1)
        # A natural distribution to choose for the weights is a dirichlet
        # distribution, which hyperparameters that need priors specified

        # return weights for different models and take logs
        weights = self.weights
        log_weights = np.log(weights + eps)

        # calculate log likelihoods
        if len(self.nargs_for_each_model) == 0:
            log_likelis = np.array(
                [log_likelihood_elementwise(model, self.x_exp, self.y_exp, self.y_err) + log_weight
                 for model, log_weight in zip(self.models, log_weights)])
        else:
            log_likelis = np.array(
                [log_likelihood_elementwise(model, self.x_exp, self.y_exp, self.y_err, params) + log_weight
                 for model, params, log_weight in zip(self.models, model_params, log_weights)])

        total_sum = np.logaddexp.reduce(log_likelis)
        return total_sum.item()

    # def mix_loglikelihood_test(self, mixture_params):
    #     W = mixture_function(self.method, self.x_exp, mixture_params)

    #     W_1 = W
    #     W_2 = 1 - W
    #     complete_array=np.append(W_1*np.exp(self.L1), W_2*np.exp(self.L2))

    #     return np.log(np.sum(complete_array)).item()

    def prediction(self, mixture_params : np.ndarray, x : np.ndarray, model_params=[]) -> np.ndarray:
        """
        predictions from mixed model for given mixing function parameters and at input values x

        Parameters
        ----------
        mixture_params : np.1darray
            parameter values that fix the shape of mixing function
        x : np.1daray
            input parameter values array
        model_params: list[np.1darray]
            list of parameter values for each model


        Returns
        ---------
        prediction : np.float
            the prediction made for mixed model
        """

        # FIXME: What if I need to return an array of predictions? Should I return a np.sum(..., axis=1)
        weights = self.weights(mixture_params, x)

        if len(self.nargs_for_each_model) == 0:
            return np.sum([weight * model.predict(x)[0]
                           for weight, model in zip(weights, self.models)])
        else:
            return np.sum([weight * model.predict(x, params)[0]
                           for weight, model, params in zip(weights, self.models, model_params)])

    def plot_weights(self, mixture_params : np.ndarray, x : np.ndarray) -> np.ndarray:
        """
        plot the mixing function against the input parameter values x

        Parameters
        ----------
        mixture_params : np.1darray
            parameter values that fix the shape of mixing function
        x : np.1darray
            input parameter values
        """

        if self.method != 'dirchlet' and self.method != 'beta':
            fig, ax = plt.subplots()
            weights = mixture_function(self.method, x, mixture_params)
            ax.plot(x, weights[0], label=self.method)
            ax.legend()
            ax.set_ylabel('Weights')
            ax.set_xlabel('Input Parameter')
        else:
            weights = mixture_function(self.method, x, mixture_params)
            fig, ax = plt.subplots()
            ax.scatter(np.arange(len(weights)), weights)
            ax.set_xlabel('Model number')
            ax.set_ylabel('Model weight')
        return None

    def weights(self, mixture_params : np.ndarray, x : np.ndarray) -> np.ndarray :
        """
        return the mixing function value at the input parameter values x

        Parameters
        ----------
        mixture_params : np.1darray
            parameter values that fix the shape of mixing function
        x : np.1darray
            input parameter values
        """

        return mixture_function(self.method, x, mixture_params)

    def posterior(self):
        pass

    def predict(self):
        pass

    def prior(self):
        pass

    def prior_predict(self):
        pass

    def set_prior(self):
        pass
