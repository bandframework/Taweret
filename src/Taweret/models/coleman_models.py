import numpy as np
from Taweret.core.base_model import BaseModel
from Taweret.utils.utils import normal_log_likelihood_elementwise as log_likelihood_elementwise_utils
from Taweret.utils.utils import normed_mvn_loglike
import bilby


class coleman_model_1(BaseModel):

    def __init__(self) -> None:
        self._prior = None

    def evaluate(
            self,
            input_values: np.array,
            model_param: np.array,
            full_corr=False) -> np.array:
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
        var = 0.3 * 0.3 * np.zeros(len(x))

        if len(model_param.flatten()) != 1:
            raise TypeError(
                'The model_param has to be single element numpy array')

        mean = 0.5 * (x + model_param.item()) - 2
        # if full_corr:
        #     # for now coleman models do not have full covariance.
        return mean.flatten(), np.sqrt(var).flatten(), None
        # else:
        #     return mean.flatten(), np.sqrt(var).flatten()

    def log_likelihood_elementwise(self, x_exp, y_exp, y_err, model_param):
        x_exp = x_exp.flatten()
        y_exp = y_exp.flatten()
        y_err = y_err.flatten()
        # log likleihood elementwise can only handle 1 observable
        return log_likelihood_elementwise_utils(
            self, x_exp, y_exp, y_err, model_param)

    # def log_likelihood(self, x_exp, y_exp_all, y_err_all, W, model_param=None):
    #     """
    # Calculate Normal log likelihood for all centrality in x_exp with
    # weights.

    #     Parameters
    #     ----------

    #     x_exp :

    #     y_exp_all :

    #     y_err_all :

    #     W :
    #     """

    #     predictions, model_errs = self.evaluate(x_exp, model_param)
    #     x_exp = x_exp.flatten()
    #     if len(x_exp)!=y_exp_all.shape[0]:
    #         raise Exception(f'Dimensionality mistmach between x_exp and y_exp')
    #     #Since the y_Exp_all has the shape of n_centralities * n_observabl_types
    #     weights = []
    #     for w in W:
    #         weights.append(w*np.ones(y_exp_all.shape[1]))
    #     weights = np.array(weights).flatten()
    #     predictions = np.array(predictions).flatten()
    #     y_exp_all = np.array(y_exp_all).flatten()
    #     y_err_all = np.array(y_err_all).flatten()
    #     diff = (predictions - y_exp_all)*weights
    #     final_cov = np.diag(np.square(y_err_all))
    #     N = len(x_exp)
    #     x_length = x_exp[1] - x_exp[0]
    #     for i in range(0,N):
    #         for j in range(0,N):
    #             rho = 0.6*x_length/((x_exp[i]-x_exp[j])**2)
    #             if i==j:
    #                 continue
    #             final_cov[i,j]=np.sqrt(final_cov[i,i]*final_cov[j,j])*rho
    #     return normed_mvn_loglike(diff,final_cov)

    def set_prior(self, bilby_priors=None):
        '''
        Set the prior on model parameters.
        '''
        if bilby_priors is None:
            print('Using default priors for model 1')
            priors = bilby.prior.PriorDict()
            priors['model1_0'] = bilby.core.prior.Uniform(1, 6, "model1_0")
        else:
            priors = bilby_priors
        print(priors)
        self._prior = priors
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

    def evaluate(
            self,
            input_values: np.array,
            model_param: np.array,
            full_corr=False) -> np.array:
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
        var = 0.3 * 0.3 * np.zeros(len(x))

        if len(model_param.flatten()) != 1:
            raise TypeError(
                'The model_param has to be single element numpy array')

        mean = -0.5 * (x - model_param.item()) + 3.7
        # if full_corr:
        #     # for now coleman models do not have full covariance.
        return mean, np.sqrt(var), None
        # else:
        #     return mean, np.sqrt(var)

    def log_likelihood_elementwise(self, x_exp, y_exp, y_err, model_param):
        x_exp = x_exp.flatten()
        y_exp = y_exp.flatten()
        y_err = y_err.flatten()
        # log likleihood elementwise can only handle 1 observable
        return log_likelihood_elementwise_utils(
            self, x_exp, y_exp, y_err, model_param)

    # def log_likelihood(self, x_exp, y_exp_all, y_err_all, W, model_param=None):
    #     """
    # Calculate Normal log likelihood for all centrality in x_exp with
    # weights.

    #     Parameters
    #     ----------

    #     x_exp :

    #     y_exp_all :

    #     y_err_all :

    #     W :
    #     """

    #     predictions, model_errs = self.evaluate(x_exp, model_param)
    #     x_exp = x_exp.flatten()
    #     if len(x_exp)!=y_exp_all.shape[0]:
    #         raise Exception(f'Dimensionality mistmach between x_exp and y_exp')
    #     #Since the y_Exp_all has the shape of n_centralities * n_observabl_types
    #     weights = []
    #     for w in W:
    #         weights.append(w*np.ones(y_exp_all.shape[1]))
    #     weights = np.array(weights).flatten()
    #     predictions = np.array(predictions).flatten()
    #     y_exp_all = np.array(y_exp_all).flatten()
    #     y_err_all = np.array(y_err_all).flatten()
    #     diff = (predictions - y_exp_all)*weights
    #     final_cov = np.diag(np.square(y_err_all))
    #     N = len(x_exp)
    #     x_length = x_exp[1] - x_exp[0]
    #     for i in range(0,N):
    #         for j in range(0,N):
    #             rho = 0.6*x_length/((x_exp[i]-x_exp[j])**2)
    #             if i==j:
    #                 continue
    #             final_cov[i,j]=np.sqrt(final_cov[i,i]*final_cov[j,j])*rho
    #     return normed_mvn_loglike(diff,final_cov)

    def set_prior(self, bilby_priors=None):
        '''
        Set the prior on model parameters.
        '''
        if bilby_priors is None:
            print('Using default priors for model 2')
            priors = bilby.prior.PriorDict()
            priors['model2_0'] = bilby.core.prior.Uniform(-2, 3, "model2_0")
        else:
            priors = bilby_priors
        print(priors)
        self._prior = priors
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

    def evaluate(self, input_values: np.array) -> np.array:
        """
        Predict the mean and error for given input values

        Parameters
        ----------
        input_values : numpy 1darray
            input parameter values
        """

        x = input_values.flatten()
        mean = np.zeros(len(x))
        var = 0.3 * 0.3 * np.ones(len(x))

        mean = 2 - 0.1 * (x - 4)**2
        return mean, np.sqrt(var)

    def log_likelihood_elementwise(self, x_exp, y_exp, y_err, model_param):
        pass

    def set_prior(self, bilby_priors=None):
        pass
