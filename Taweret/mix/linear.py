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

    def __init__(self, model_1, model_2, x_exp, y_exp, y_err, method='sigmoid',n_model_1=0\
        ,n_model_2=0, n_mix=0):
        """
        Parameters
        ----------
        model_1 : object with a predict method
            first model to be mixed
        model_2 : object with a predict method
            second model to be mixed
        x_exp : np.1darray
            input parameter values of experimental data
        y_exp : np.1darray
            mean of the experimental data for each input value
        y_err : np.1darray
            standard deviation of the experimental data for each input value
        method : str
            mixing function
        n_model_1 : int
            number of free parameters in the model 1
        n_model_2 : int
            number of free parameters in the model 2
        n_mix : int
            number of free parameters in the mixing funtion
        """

        #check for predict method in the models
        try:
            getattr(model_1, 'predict')
        except AttributeError:
            print('model 1 does not have a predict method')
        else:
            self.model_1 = model_1

        try:
            getattr(model_2, 'predict')
        except AttributeError:
            print('model 2 does not have a predict method')
        else:
            self.model_2 = model_2
        

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

        self.n_model_1 = n_model_1
        self.n_model_2 = n_model_2
        self.n_mix = n_mix


        # Calculate log likelihood of each model
        if self.n_model_1==0 and self.n_model_2==0:

            self.L1 = log_likelihood_elementwise(model_1,x_exp, y_exp, y_err)
            self.L2 = log_likelihood_elementwise(model_2,x_exp, y_exp, y_err)
        #print(self.L1)
        #print(self.L2)

    def mix_loglikelihood(self, mixture_params : np.ndarray, model_1_param=np.array([]), model_2_param=np.array([])) -> float:
        """
        log likelihood of the mixed model given the mixing function parameters

        Parameters
        ----------
        mixture_params : np.1darray
            parameter values that fix the shape of mixing function
        model_1_param: np.1darray
            parameter values in the model 1
        model_2_param: np.1darray
            parameter values  in the model 2
        """

        W = mixture_function(self.method, self.x_exp, mixture_params)
        W_1 = np.log(W + eps)
        W_2 = np.log(1 - W + eps)
        if self.n_model_1==0 and self.n_model_2==0:
            complete_array=np.append(W_1+self.L1, W_2+self.L2)
        else:
            L1 = log_likelihood_elementwise(self.model_1, self.x_exp, self.y_exp, self.y_err, model_1_param)
            L2 = log_likelihood_elementwise(self.model_2, self.x_exp, self.y_exp, self.y_err, model_2_param)
            complete_array=np.append(W_1+L1, W_2+L2)
        #print(complete_array)
        total_sum = 0
        for i in range(0,len(complete_array)-1):
            if i==0:
                total_sum=np.logaddexp(complete_array[i],complete_array[i+1])
            else:
                total_sum=np.logaddexp(total_sum, complete_array[i+1])
            #print(total_sum)
        return total_sum.item()

    # def mix_loglikelihood_test(self, mixture_params):
    #     W = mixture_function(self.method, self.x_exp, mixture_params)
        
    #     W_1 = W
    #     W_2 = 1 - W
    #     complete_array=np.append(W_1*np.exp(self.L1), W_2*np.exp(self.L2))

    #     return np.log(np.sum(complete_array)).item()

    def prediction(self, mixture_params : np.ndarray, x : np.ndarray, model_1_param=np.array([]), model_2_param=np.array([])) -> np.ndarray:
        """
        predictions from mixed model for given mixing function parameters and at input values x

        Parameters
        ----------
        mixture_params : np.1darray
            parameter values that fix the shape of mixing function
        x : np.1daray
            input parameter values array
        model_1_param: np.1darray
            parameter values in the model 1
        model_2_param: np.1darray
            parameter values  in the model 2
        """
        w = mixture_function(self.method, x, mixture_params)

        if self.n_model_1==0 and self.n_model_2==0:
            return w * self.model_1.predict(x)[0] + (1-w) * self.model_2.predict(x)[0]
        else:
            return w * self.model_1.predict(x, model_1_param)[0] + (1-w) * self.model_2.predict(x, model_2_param)[0]

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

        fig, ax = plt.subplots()
        ax.plot(x, mixture_function(self.method, x, mixture_params), label=self.method)
        ax.set_ylabel('Weights')
        ax.set_xlabel('Input Parameter')
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


