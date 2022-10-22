# This will have all the linear bayesian model mixing methods.
# Takes Models as inputs:
# Check if Models have an predict method and they should output a mean and a variance.

import numpy as np
import math
from ..utils.utils import log_likelihood_elementwise, mixture_function, eps
import matplotlib.pyplot as plt

class linear_mix():
    """
    Generates a linear mixed model

    """

    def __init__(self, models, x_exp, y_exp, y_err, method='sigmoid',
                 nargs_for_each_model=[], n_mix=0):
        """
        Parameters
        ----------
        models : list
            models to mix, each must contain a predict method
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
            raise Exception('in linear_mix.__init__: len(models) must either equal len(nargs_for_each_model) or 0')

        #check for predict method in the models
        for i, model in enumerate(models):
            try:
                getattr(model, 'predict')
            except AttributeError:
                print(f'model {i} does not have a predict method')
            else:
                continue

        self.models = models

        #check if the dimensions match for experimental data
        if (x_exp.shape != y_exp.shape) or (x_exp.shape!=y_err.shape):
            raise Exception('x_exp, y_exp, y_err all should have the same dimensions')

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

        # check that list of models and mixture_params has same length
        if len(self.models) != len(mixture_params) and len(self.nargs_for_each_model) != 0:
            raise Exception('linear_mix.mix_loglikelihood: mixture_params has wrong length')

        # here, the mixture_function should support dirchlet distribution which is the most
        # general distribution for the symplex defined be mixture_params

        # return weights for different models and take logs
        weights = mixture_function(self.method, self.x_exp, mixture_params)
        log_weights = np.array([np.log(weight + eps) for i, weight in enumerate(weights)])

        # calculate log likelihoods
        if len(self.nargs_for_each_model) == 0:
            log_likelis = np.array(
                [log_likelihood_elementwise(model, self.x_exp, self.y_exp, self.y_err) + log_weight
                 for model, log_weight in zip(self.models, log_weights)])
        else:
            log_likelis = np.array(
                [log_likelihood_elementwise(model, self.x_exp, self.y_exp, self.y_err, params) + log_weight
                 for model, params, log_weight in zip(self.models, model_params, log_weights)])

        total_sum = np.log(np.sum(np.exp(log_likelis)))
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
        weights = mixture_function(self.method, x, mixture_params)

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


