# This will have all the linear bayesian model mixing methods.
# Takes Models as inputs:
# Check if Models have an predict method and they should output a mean and a variance.
#
# Modified by K. Ingles

import _mypackage

from Taweret.core.base_mixer import BaseMixer
from Taweret.core.base_model import BaseModel
from Taweret.utils.utils import log_of_normal_dist

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm, dirichlet
from typing import Any, Dict, List, Optional

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
        if (
            len(models) != len(nargs_for_each_model)
            and len(nargs_for_each_model) != 0
        ):
            raise Exception(
                "in linear_mix.__init__: len(nargs_for_each_model) must either equal len(models) or 0"
            )

        # check for predict method in the models
        for i, model in enumerate(models):
            try:
                issubclass(type(model), BaseModel)
            except AttributeError:
                print(
                    f"model {i} needs to inherit from Taweret.core.base_model.BaseModel"
                )
            else:
                continue

        self.models = models
        self.nargs_for_each_model = nargs_for_each_model
        self.n_mix = n_mix

        # function returns

    def evaluate(
        self, mixing_params: np.ndarray, model_params: Dict[str, List[Any]]
    ) -> np.ndarray:
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

        weights = self.sample_prior(num_samples=1)

        if len(self.nargs_for_each_model) == 0:
            return np.sum(
                [
                    weight * model.evaluate()
                    for weight, model in zip(weights, self.models.value())
                ]
            )
        else:
            return np.sum(
                [
                    weight * model.evaluate(*params)
                    for weight, model, params in zip(
                        weights, self.models.values(), model_params
                    )
                ]
            )

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

    @property
    def map(self):
        if self.has_trained:
            return self.m_map
        else:
            raise Exception("Please train model before requesting MAP")

    def mix_loglikelihood(
        self, y_exp, y_err, mix_params=[], model_params=None
    ) -> float:
        """
        log likelihood of the mixed model given the mixing function parameters

        Parameters
        ----------
        mixture_params : np.1darray
            parameter values that fix the shape of mixing function
        model_params: list[np.1darray]
            list of model parameters for each model, note that different models
            can take different number of params
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
                [
                    log_of_normal_dist(model.evaluate(), y_exp, y_err)
                    + log_weight
                    for model, log_weight in zip(self.models, log_weights)
                ]
            )
        else:
            log_likelis = np.array(
                [
                    log_of_normal_dist(model.evaluate(*params), y_exp, y_err)
                    + log_weight
                    for model, params, log_weight in zip(
                        self.models, model_params, log_weights
                    )
                ]
            )

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
        ax.set_xlabel("Model number")
        ax.set_ylabel("Model weight")
        return None

    @property
    def posterior(self):
        if self.has_trained:
            return self.m_posterior
        else:
            raise Exception("Please train model before requesting posterior")

    def predict(self):
        if self.has_trained:
            return None
        else:
            raise Exception("Please train model before making predictions")

    @property
    def prior(self):
        if self.m_prior is not None:
            return self.m_prior
        else:
            print("No priors have been set")
            return None

    def prior_predict(
        self, num_samples: int, model_params: Dict[str, List[float]]
    ) -> np.ndarray:
        prior_points = self.sample_prior(num_samples=num_samples)
        print(prior_points.shape)
        if len(self.nargs_for_each_model) == 0:
            return np.array(
                [
                    np.sum(
                        [
                            prior * model.evaluate()
                            for prior, model in zip(prior_point, self.models)
                        ]
                    )
                    for prior_point in prior_points
                ]
            )
        else:
            return np.ndarray(
                [
                    np.sum(
                        [
                            prior * model.evaluate(*params)
                            for prior, params, model in zip(
                                prior_point, model_params, self.models
                            )
                        ]
                    )
                    for prior_point in prior_points
                ]
            )

    def predict_weights(self, num_samples: int) -> np.ndarray:
        if self.has_trained:
            return None
        else:
            raise Exception("Please train model before making predictions")

    def sample_prior(self, num_samples: int) -> np.ndarray:
        # 1. Sample log-normal distribution to get log-hyperparameters
        # 2. Feed exponeniates random variates to a dirichlet instance and sample
        # 3. Return array with samples from dirchilet distribution
        prior_samples = []
        for n in np.arange(num_samples):
            log_norm_samples = np.array(
                [dist.rvs() for dist in self.m_prior.values()]
            )
            prior_samples.append(dirichlet(np.exp(log_norm_samples)).rvs())

        prior_samples = np.vstack(prior_samples)
        if num_samples == 1:
            return prior_samples[0]
        else:
            return prior_samples

    def set_prior(self):
        print("Notice: For global fitting, the prior is always assumed to be")
        print("        a Dirichlet distribution")

        scale = 1
        prior_dict = {f"param_{i}": lognorm(scale) for i in range(self.n_mix)}
        self.m_prior = prior_dict

    def _loglikelihood_for_sampler(
        self, x: Any, y_exp: np.ndarray, y_err: np.ndarray, model_params=None
    ) -> np.ndarray:
        return self.mix_loglikelihood(
            y_exp=y_exp, y_err=y_err, mix_params=x, model_params=model_params
        )

    def _log_prior(self, prior_params: np.ndarray) -> np.ndarray:
        return np.sum(
            [
                prior.pdf(param)
                for prior, param in zip(self.m_prior.values(), prior_params)
            ]
        )

    def train(
        self,
        y_exp: np.ndarray,
        y_err: np.ndarray,
        model_params: Optional[Dict[str, List[float]]],
    ):
        import ptemcee

        nargs = (
            np.max(np.asarray(self.nargs_for_each_model.values()))
            if len(self.nargs_for_each_model) != 0
            else len(self.models)
        )
        # TODO: Should these be function parameters?
        nsteps = 2000 * nargs
        nburn = 100 * nargs
        ntemps = 10
        nwalkers = 20 * nargs

        starting_guess = self.sample_prior(num_samples=1)
        print(starting_guess)
        sampler = ptemcee.Sampler(
            nwalkers=nwalkers,
            dim=self.n_mix,
            ntemps=ntemps,
            Tmax=10,
            threads=4,
            logl=self._loglikelihood_for_sampler,
            logp=self._log_prior,
            loglargs=[y_exp, y_err, model_params],
        )
        print("Burn-in sampling")
        x = sampler.run_mcmc(
            p0=starting_guess, iterations=nburn, swap_ratios=True
        )
        print("Burn-in samping complete")
        sampler.rest()
        print("Now running other samples")
        x = sampler.run_mcmc(
            p0=x[0], iterations=nsteps, storechain=True, swap_ratios=True
        )

        self.m_posterior = np.array(sampler.chain)
        self.evidence = sampler.log_evidence_estimate()

        return self.m_posterior


class LinearMixerLocal(BaseMixer):
    """
    Generates a linear mixed model

    """

    def __init__(
        self,
        models,
        x_exp,
        y_exp,
        y_err,
        method="sigmoid",
        nargs_for_each_model=[],
        n_mix=0,
    ):
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
        if (
            len(models) != len(nargs_for_each_model)
            and len(nargs_for_each_model) != 0
        ):
            raise Exception(
                "in linear_mix.__init__: len(nargs_for_each_model) must either equal len(models) or 0"
            )

        # check for predict method in the models
        for i, model in enumerate(models):
            try:
                issubclass(type(model), BaseModel)
            except AttributeError:
                print(
                    f"model {i} needs to inherit from Taweret.core.base_model.BaseModel"
                )
            else:
                continue

        self.models = models

        # check if the dimensions match for experimental data
        if (x_exp.shape[0] != y_exp.shape[0]) or (
            x_exp.shape[0] != y_err.shape[0]
        ):
            raise Exception(
                "x_exp, y_exp, y_err all should have the same shape[0]"
            )

        self.x_exp = x_exp.flatten()
        self.y_exp = y_exp.flatten()
        self.y_err = y_err.flatten()

        # check if mixing method exist
        if method not in ["step", "sigmoid", "cdf"]:
            raise Exception(
                "only supports the step or sigmoid mixing functions"
            )

        self.method = method

        self.nargs_for_each_model = nargs_for_each_model
        self.n_mix = n_mix

        # function returns

    def mix_loglikelihood(
        self, mixture_params: np.ndarray, model_params=[]
    ) -> float:
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
                [
                    log_likelihood_elementwise(
                        model, self.x_exp, self.y_exp, self.y_err
                    )
                    + log_weight
                    for model, log_weight in zip(self.models, log_weights)
                ]
            )
        else:
            log_likelis = np.array(
                [
                    log_likelihood_elementwise(
                        model, self.x_exp, self.y_exp, self.y_err, params
                    )
                    + log_weight
                    for model, params, log_weight in zip(
                        self.models, model_params, log_weights
                    )
                ]
            )

        total_sum = np.logaddexp.reduce(log_likelis)
        return total_sum.item()

    # def mix_loglikelihood_test(self, mixture_params):
    #     W = mixture_function(self.method, self.x_exp, mixture_params)

    #     W_1 = W
    #     W_2 = 1 - W
    #     complete_array=np.append(W_1*np.exp(self.L1), W_2*np.exp(self.L2))

    #     return np.log(np.sum(complete_array)).item()

    def prediction(
        self, mixture_params: np.ndarray, x: np.ndarray, model_params=[]
    ) -> np.ndarray:
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
            return np.sum(
                [
                    weight * model.predict(x)[0]
                    for weight, model in zip(weights, self.models)
                ]
            )
        else:
            return np.sum(
                [
                    weight * model.predict(x, params)[0]
                    for weight, model, params in zip(
                        weights, self.models, model_params
                    )
                ]
            )

    def plot_weights(
        self, mixture_params: np.ndarray, x: np.ndarray
    ) -> np.ndarray:
        """
        plot the mixing function against the input parameter values x

        Parameters
        ----------
        mixture_params : np.1darray
            parameter values that fix the shape of mixing function
        x : np.1darray
            input parameter values
        """

        if self.method != "dirchlet" and self.method != "beta":
            fig, ax = plt.subplots()
            weights = mixture_function(self.method, x, mixture_params)
            ax.plot(x, weights[0], label=self.method)
            ax.legend()
            ax.set_ylabel("Weights")
            ax.set_xlabel("Input Parameter")
        else:
            weights = mixture_function(self.method, x, mixture_params)
            fig, ax = plt.subplots()
            ax.scatter(np.arange(len(weights)), weights)
            ax.set_xlabel("Model number")
            ax.set_ylabel("Model weight")
        return None

    def weights(self, mixture_params: np.ndarray, x: np.ndarray) -> np.ndarray:
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
