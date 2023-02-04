# This will have all the linear bayesian model mixing methods.
# Takes Models as inputs:
# Check if Models have an predict method and they should output a mean and a variance.
#
# Modified by K. Ingles

import _mypackage

from Taweret.core.base_mixer import BaseMixer
from Taweret.core.base_model import BaseModel

import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import cpu_count
from scipy.stats import norm, dirichlet
from typing import Any, Dict, List, Optional

from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process.kernels import RBF

eps = 1.0e-20


class LinearMixerGlobal(BaseMixer):
    """
    Generates a global linear mixed model

    """

    def __init__(self, models, nargs_for_each_model=None, n_mix=0):
        """
        Parameters
        ----------
        models : Dict[str, Model(BaseModel)]
            models to mix, each must contain a evaluate method
        nargs_for_each_model : Dict[str, int]
            number of free parameters for each model
        n_mix : int
            number of free parameters in the mixing funtion
        """

        # check that lengths of lists are compatible
        if not (nargs_for_each_model is None):
            if len(models) != len(nargs_for_each_model):
                raise Exception(
                    "in linear_mix.__init__: len(nargs_for_each_model)"
                    + "must either equal len(models) or 0"
                )

        # check for predict method in the models
        for key, model in models.items():
            try:
                issubclass(type(model), BaseModel)
            except AttributeError:
                print(
                    f"model {key} needs to inherit from " +
                    "Taweret.core.base_model.BaseModel"
                )
            else:
                continue

        self.models = models
        self.nargs_for_each_model = nargs_for_each_model
        self.n_mix = n_mix
        self.has_trained = False

        # function returns

    ##########################################################################
    ##########################################################################

    # TODO: This function seems to contradict our philosophy
    def evaluate(
        self,
        mixing_parameters: np.ndarray,
        model_parameters: Optional[Dict[str, List[float]]] = None,
    ) -> np.ndarray:
        """
        evaluate mixed model for given mixing function parameters

        Parameters
        ----------
        mixture_parameters : np.ndarray
            parameter values that fix the shape of mixing function
        model_parameters : Optional[Dict[str, List[float]]]
            Dictionary with list of parameter values for each model


        Returns
        -------
        evaluation : float
            evaluation of the mixing model
        """

        # FIXME: I am currently returning the weights, but I should be
        #        sampling the log-normal distribution for the dirichlet
        #        hyperparameters
        weights = self.evaluate_weights(self._sample_prior(number_samples=1))

        if len(self.nargs_for_each_model) == 0:
            return np.sum(
                [
                    weight * model.evaluate()
                    for weight, model in zip(weights, self.models.values())
                ]
            )
        else:
            return np.sum(
                [
                    weight * model.evaluate(*parameters)
                    for weight, model, parameters in zip(
                        weights,
                        self.models.values(),
                        model_parameters.values(),
                    )
                ]
            )

    ##########################################################################
    ##########################################################################

    def evaluate_weights(self, mix_parameters: np.ndarray) -> np.ndarray:
        """
        calculate the weights given some set of input parameters

        Parameters
        ----------
        mixture_parameters : np.1darray
            parameter values that fix the shape of mixing function

        Returns:
        --------
        weights : np.ndarray
            array of sampled weights
        """
        n_attempts_allowed = 100
        n_attempts = 0
        while True and n_attempts < n_attempts_allowed:
            sample = dirichlet(np.exp(mix_parameters)).rvs()
            if not np.any(np.isnan(sample)):
                return sample[0]
            n_attempts += 1
        raise Exception("Too many NaNs in sampling Dirichlet distribution")

    ##########################################################################
    ##########################################################################

    @property
    def map(self):
        """
        Maximum a posterior value from posterior
        """
        if self.has_trained:
            return self.m_map
        else:
            raise Exception("Please train model before requesting MAP")

    ##########################################################################
    ##########################################################################

    def mix_loglikelihood(
        self,
        y_exp: np.ndarray,
        y_err: np.ndarray,
        mix_parameters: np.ndarray,
        model_parameters: Optional[Dict[str, List[float]]] = None,
    ) -> float:
        """
        log likelihood of the mixing model

        Parameters
        ----------
        y_exp : np.ndarray
            The experimental data
        y_err : np.ndarray
            Gaussian error bars on the experimental data
        mix_parameters : np.1darray
            parameter values that fix the shape of mixing function
        model_parameters : Optional[Dict[str, List[float]]]
            list of model parameters for each model, note that different models
            can take different number of parameters

        Returns:
        --------
        log_likelihood : float
            log_likelihood of the model given map parameters for the models and
            a set of weights
        """

        # In general, the weights statisfy a simplex condition (they sum to 1)
        # A natural distribution to choose for the weights is a dirichlet
        # distribution, which hyperparameters that need priors specified

        # return weights for different models and take logs
        weights = self.evaluate_weights(mix_parameters)
        log_weights = np.log(weights + eps)

        # calculate log likelihoods
        if model_parameters is None:
            log_likelis = np.array(
                [
                    model.log_likelihood_elementwise(y_exp, y_err) + log_weight
                    for model, log_weight in zip(
                        self.models.values(), log_weights
                    )
                ]
            )
        else:
            log_likelis = np.array(
                [
                    model.log_likelihood_elementwise(y_exp, y_err, *parameters)
                    + log_weight
                    for model, parameters, log_weight in zip(
                        self.models.values(),
                        model_parameters.values(),
                        log_weights,
                    )
                ]
            )

        mix_log_likeli = np.logaddexp.reduce(log_likelis)
        return mix_log_likeli.item()

    ##########################################################################
    ##########################################################################

    def plot_weights(self, mix_parameters: np.ndarray) -> np.ndarray:
        """
        plot the mixing function against the input parameter values x

        Parameters
        ----------
        mixture_parameters : np.1darray
            parameter values that fix the shape of mixing function

        Returns:
        --------
        None
        """
        weights = self.evaluate_weights(mix_parameters)
        fig, ax = plt.subplots()
        ax.scatter(np.arange(len(weights)), weights)
        ax.set_xlabel("Model number")
        ax.set_ylabel("Model weight")
        return None

    ##########################################################################
    ##########################################################################

    @property
    def posterior(self):
        """
        Stores the most recent posteriors from running self.train function

        Returns:
        --------
        self.m_posterior : np.ndarray
            posterior from learning the weights
        """
        if self.has_trained:
            return self.m_posterior
        else:
            raise Exception("Please train model before requesting posterior")

    ##########################################################################
    ##########################################################################

    def _sample_distribution(
        self,
        distribution: np.ndarray,
        model_parameters: Optional[Dict[str, List[float]]] = None,
    ) -> np.ndarray:
        """
        Helper function to evaluate the predictive distribution from a given
        posterior distribution

        Parameters:
        -----------
        distribution : np.ndarray
            Can be the MCMC chain from the posterior sampling, or a set of
            sample points selected by the user
        model_parameters : Optional[Dict[str, List[float]]]
            dictionary contain lists of the parameters each model needs to use

        Returns:
        --------
        sample : np.ndarray
            array of points where mixing model was evaluated at
        """
        return np.array(
            [
                self.evaluate(sample, model_parameters)
                for sample in distribution.reshape(-1, self.n_mix)
            ]
        )

    ##########################################################################
    ##########################################################################

    def predict(
        self,
        model_parameters: Optional[Dict[str, List[float]]] = None,
        credible_intervals=[5, 95],
        samples=None,
    ):
        """
        Evaluate posterior to make prediction at test points.

        Parameters:
        -----------
        model_parameters : Optional[Dict[str, List[float]]]
            dictionary contain lists of the parameters each model needs to use
        credible_intervals : Optional[List[float], List[List[float]]]
            list of even number of integers the express which percentiles of
            the posterior distribution to compute and return
        sample : np.ndarray
            User supplied distribution, to use instead of the existing
            distribution

        Returns:
        --------
        evaluated_posterior : np.ndarray
            array of posterior predictive distribution evaluated at provided
            test points
        credible_intervals : np.ndarray
            intervals corresponding for 60%, 90% credible intervals
        mean : np.ndarray
            average mixed model value at each provided test points
        std_dev : np.ndarray
            sample standard deviation of mixed model output at provided test
            points
        """

        if self.has_trained and samples is None:
            predictive_distribution = self._sample_distribution(
                self.m_posterior, model_parameters
            )

            return_intervals = np.percentile(
                predictive_distribution, np.asarray(credible_intervals), axis=0
            )
            return_mean = np.mean(predictive_distribution)
            return_stddev = np.std(predictive_distribution)
            return (
                predictive_distribution,
                return_intervals,
                return_mean,
                return_stddev,
            )
        else:
            if samples is None:
                raise Exception(
                    "Please either train model, or provide samples as an" +
                    "argument"
                )
            else:
                predictive_distribution = self._sample_distribution(
                    samples, model_parameters
                )

                return_intervals = np.percentile(
                    predictive_distribution, credible_intervals
                )
                return_mean = np.mean(predictive_distribution)
                return_stddev = np.std(predictive_distribution)
                return (
                    predictive_distribution,
                    return_intervals,
                    return_mean,
                    return_stddev,
                )

            raise Exception("Please train model before making predictions")

    ##########################################################################
    ##########################################################################

    @property
    def prior(self):
        """
        Dictionary of prior distributions. Format should be compatible with
        sampler.

        Returns:
        --------
        self.m_prior : Dict[str, Any]
            Underlying prior object(s)
        """
        if self.m_prior is not None:
            return self.m_prior
        else:
            print("No priors have been set")
            return None

    ##########################################################################
    ##########################################################################

    def prior_predict(
        self,
        model_parameters: Optional[Dict[str, List[float]]] = None,
        credible_interval=[5, 95],
        number_samples: int = 1000,
    ) -> np.ndarray:
        """
        Get prior predictive distribution and prior distribution samples

        Parameters:
        -----------
        model_parameters : Optional[Dict[str, List[float]]]
            dictionary contain lists of the parameters each model needs to use
        credible_intervals : Optional[List[float], List[List[float]]]
            list of even number of integers the express which percentiles of
            the posterior distribution to compute and return
        sample : np.ndarray
            User supplied distribution, to use instead of the existing
            distribution

        Returns:
        --------
        evaluated_prior : np.ndarray
            array of prior predictive distribution evaluated at provided
            test points
        credible_intervals : np.ndarray
            intervals corresponding for 60%, 90% credible intervals
        mean : np.ndarray
            average mixed model value at each provided test points
        std_dev : np.ndarray
            sample standard deviation of mixed model output at provided test
            points
        """

        prior_points = self._sample_prior(number_samples=number_samples)
        prior_points = np.exp(prior_points)
        prior_points = np.array(
            [dirichlet(prior_point).rvs() for prior_point in prior_points]
        )
        return self.predict(
            model_parameters=model_parameters,
            credible_interval=credible_interval,
            samples=prior_points,
        )

    ##########################################################################
    ##########################################################################

    def predict_weights(
        self, credible_interval=[5, 95], samples=None
    ) -> np.ndarray:
        """
        Calculate posterior predictive distribution for model weights

        Parameters:
        -----------
        credible_intervals : Optional[List[float], List[List[float]]]
            list of even number of integers the express which percentiles of
            the posterior distribution to compute and return
        sample : np.ndarray
            User supplied distribution, to use instead of the existing
            distribution

        Returns:
        --------
        evaluated_posterior : np.ndarray
            array of posterior predictive distribution evaluated at provided
            test points
        mean : np.ndarray
            average mixed model value at each provided test points
        credible_intervals : np.ndarray
            intervals corresponding for 60%, 90% credible intervals
        std_dev : np.ndarray
            sample standard deviation of mixed model output at provided test
            points
        """

        if self.has_trained:
            return np.array(
                [
                    self.evaluate_weights(sample)
                    for sample in self.m_posterior.reshape(-1, self.n_mix)
                ]
            )
        else:
            raise Exception("Please train model before making predictions")

    ##########################################################################
    ##########################################################################

    def _sample_prior(self, number_samples: int) -> np.ndarray:
        """
        Helper function to sample prior since all prior distributions are
        stochastic in nature

        Parameters:
        -----------
        number_samples : int
            number of samples form prior distributions

        Returns:
        --------
        samples : np.ndarray
            array of samples with the shape (number_samples, self.n_mix)
        """
        prior_samples = []
        for n in np.arange(number_samples):
            log_norm_samples = np.array(
                [dist.rvs() for dist in self.m_prior.values()]
            )
            prior_samples.append(log_norm_samples)

        prior_samples = np.vstack(prior_samples)
        if number_samples == 1:
            return prior_samples[0]
        else:
            return prior_samples

    ##########################################################################
    ##########################################################################

    def set_prior(self, scale: float):
        """
        A call to this function automatically sets up a dictionary of length
        self.n_mix where the keys are generic strings and the values a
        lognormal distribution
        """
        print("Notice: For global fitting, the prior is always assumed to be")
        print("        a Dirichlet distribution")

        prior_dict = {
            f"param_{i}": norm(loc=0, scale=scale) for i in range(self.n_mix)
        }
        self.m_prior = prior_dict

    ##########################################################################
    ##########################################################################

    def _loglikelihood_for_sampler(
        self,
        mix_parameters: Any,
        y_exp: np.ndarray,
        y_err: np.ndarray,
        model_parameters: Optional[Dict[str, List[float]]] = None,
    ) -> np.ndarray:
        """
        Helper for ptemcee call fo loglikelihood function
        Simply rearranges the order of the loglikelihood function

        Parameters
        ----------
        mix_parameters : np.1darray
            parameter values that fix the shape of mixing function
        y_exp : np.ndarray
            The experimental data
        y_err : np.ndarray
            Gaussian error bars on the experimental data
        model_parameters : Optional[Dict[str, List[float]]]
            list of model parameters for each model, note that different models
            can take different number of parameters

        Returns:
        --------
        log_likelihood : float
            log_likelihood of the model given map parameters for the models and
            a set of weights
        """
        return self.mix_loglikelihood(
            y_exp=y_exp,
            y_err=y_err,
            mix_parameters=mix_parameters,
            model_parameters=model_parameters,
        )

    ##########################################################################
    ##########################################################################

    def _log_prior(self, prior_parameters: np.ndarray) -> np.ndarray:
        """
        Helper for ptemcee call fo log-prior function
        Simply rearranges the order of the log-prior function

        Parameters:
        -----------
        prior_parameteres : np.ndarray
            parameters to pass to prior distribution and sample it

        Returns:
        --------
        log_prior : float
            The prior distriution evaluated at the provide prior_parameters
            point(s)
        """
        return np.sum(
            [
                prior.logpdf(param)
                for prior, param in zip(
                    self.m_prior.values(), prior_parameters
                )
            ]
        )

    ##########################################################################
    ##########################################################################

    def train(
        self,
        y_exp: np.ndarray,
        y_err: np.ndarray,
        model_parameters: Optional[Dict[str, List[float]]] = None,
        steps: int = 2000,
        burn: int = 50,
        temps: int = 10,
        walkers: int = 20,
        thinning: int = 100,
    ):
        """
        Run sampler to learn weights. Method should also create class
        members that store the posterior and other diagnostic quantities
        import for plotting
        MAP values should also caluclate and set as member variable of
        class

        Parameters:
        -----------
        y_exp : np.ndarray
            experimental observables to compare models with
        y_err : np.ndarray
            gaussian error bars on observables
        model_parameters : Optional[Dict[str, List[float]]]
            dictionary which contains list of model parameters for each model
        steps: int
            Number of steps for MCMC per model (defaults to 2000)
        burn: int
            Number of burn-in steps for MCMC per model (defaults to 50)
        temps: int
            Number of temperatures to use for parallel termpering (defaults
            to 10)
        walkers: int
            Number of walkers per model (defaults to 20)
        thinning: int
            Keep every `thinning`-th step from MCMC run

        Return:
        -------
        self.m_posterior : np.ndarray
            the mcmc chain return from sampler
        """
        import ptemcee

        nsteps = steps * self.n_mix
        nburn = burn * self.n_mix
        ntemps = temps
        nwalkers = walkers * self.n_mix

        starting_guess = np.array(
            [
                self._sample_prior(number_samples=nwalkers)
                for _ in range(ntemps)
            ]
        )
        sampler = ptemcee.Sampler(
            nwalkers=nwalkers,
            dim=self.n_mix,
            ntemps=ntemps,
            Tmax=10,
            threads=cpu_count(),
            logl=self._loglikelihood_for_sampler,
            logp=self._log_prior,
            loglargs=[y_exp, y_err, model_parameters],
            logpargs=[],
        )
        print("Burn-in sampling")
        x = sampler.run_mcmc(
            p0=starting_guess, iterations=nburn, swap_ratios=True
        )
        print("Burn-in samping complete")
        sampler.reset()
        print("Now running other samples")
        x = sampler.run_mcmc(
            p0=x[0],
            iterations=nsteps,
            thin=thinning,
            storechain=True,
            swap_ratios=True
        )

        # We want the zero temperature chain
        # Recall that the shape of the chain will be:
        #   (ntemps, nwalkers, nsteps, nvars)
        self.m_posterior = np.array(sampler.chain[0, ...])
        self.evidence = sampler.log_evidence_estimate()

        del sampler
        # FIXME: This is wrong!
        self.m_map = self.m_posterior[np.argmax(self.m_posterior, axis=2)]

        self.has_trained = True
        return self.m_posterior


##########################################################################
##########################################################################
##########################################################################


class LinearMixerLocal(BaseMixer):
    """
    Generates a local linear mixed model

    """

    def __init__(self, models, nargs_for_each_model=None, n_mix=0):
        """
        Parameters
        ----------
        models : Dict[str, Model(BaseModel)]
            models to mix, each must contain a evaluate method
        nargs_for_each_model : Dict[str, int]
            number of free parameters for each model
        n_mix : int
            number of free parameters in the mixing funtion
        """

        # check that lengths of lists are compatible
        if not (nargs_for_each_model is None):
            if len(models) != len(nargs_for_each_model):
                raise Exception(
                    "in linear_mix.__init__: len(nargs_for_each_model)"
                    + "must either equal len(models) or 0"
                )

        # check for predict method in the models
        for key, model in models.items():
            try:
                issubclass(type(model), BaseModel)
            except AttributeError:
                print(
                    f"model {key} needs to inherit from" +
                    "Taweret.core.base_model.BaseModel"
                )
            else:
                continue

        self.models = models
        self.nargs_for_each_model = nargs_for_each_model
        self.n_mix = n_mix
        self.has_trained = False

        # function returns

    ##########################################################################
    ##########################################################################

    # TODO: This function seems to contradict our philosophy
    def evaluate(
        self,
        local_parameters: np.ndarray,
        model_parameters: Optional[Dict[str, List[float]]] = None,
    ) -> np.ndarray:
        """
        evaluate mixed model for given mixing function parameters

        Parameters
        ----------
        local_parameters : np.ndarray
            parameters that determine where to sample the prior distributions
        model_parameters : Optional[Dict[str, List[float]]]
            Dictionary with list of parameter values for each model


        Returns
        -------
        evaluation : float
            evaluation of the mixing model
        """

        # FIXME: I am currently returning the weights, but I should be
        #        sampling the log-normal distribution for the dirichlet
        #        hyperparameters
        weights = self.evaluate_weights(
            local_parmeters=local_parameters,
            number_samples=1
        )

        if len(self.nargs_for_each_model) == 0:
            return np.sum(
                [
                    weight * model.evaluate()
                    for weight, model in zip(weights, self.models.values())
                ]
            )
        else:
            return np.sum(
                [
                    weight * model.evaluate(*parameters)
                    for weight, model, parameters in zip(
                        weights,
                        self.models.values(),
                        model_parameters.values(),
                    )
                ]
            )

    ##########################################################################
    ##########################################################################

    def evaluate_weights(
        self,
        local_parameters: np.ndarray,
        number_samples: int = 1
    ) -> np.ndarray:
        """
        calculate the weights given some set of input parameters

        Parameters
        ----------
        local_parameters : np.ndarray
            parameters that determine where to sample the prior distribution
        number_samples : int
            number of samples to return from prior

        Returns:
        --------
        weights : np.ndarray
            array of sampled weights
        """
        shape_parameters = self._sample_prior(
            local_parmeters=local_parameters,
            number_samples=number_samples
        )

        n_attempts_allowed = 100
        n_attempts = 0
        while True and n_attempts < n_attempts_allowed:
            sample = dirichlet(np.exp(shape_parameters)).rvs()
            if not np.any(np.isnan(sample)):
                return sample[0]
            n_attempts += 1
        raise Exception("Too many NaNs in sampling Dirichlet distribution")

    ##########################################################################
    ##########################################################################

    @property
    def map(self):
        """
        Maximum a posterior value from posterior
        """
        if self.has_trained:
            return self.m_map
        else:
            raise Exception("Please train model before requesting MAP")

    ##########################################################################
    ##########################################################################

    def mix_loglikelihood(
        self,
        y_exp: np.ndarray,
        y_err: np.ndarray,
        local_parameters: np.ndarray,
        model_parameters: Optional[Dict[str, List[float]]] = None,
    ) -> float:
        """
        log likelihood of the mixing model

        Parameters
        ----------
        y_exp : np.ndarray
            The experimental data
        y_err : np.ndarray
            Gaussian error bars on the experimental data
        mix_parameters : np.1darray
            parameter values that fix the shape of mixing function
        local_parameter : np.ndarray
            parameters that determine where to sample the prior
        model_parameters : Optional[Dict[str, List[float]]]
            list of model parameters for each model, note that different models
            can take different number of parameters

        Returns:
        --------
        log_likelihood : float
            log_likelihood of the model given map parameters for the models and
            a set of weights
        """

        # In general, the weights statisfy a simplex condition (they sum to 1)
        # A natural distribution to choose for the weights is a dirichlet
        # distribution, which hyperparameters that need priors specified

        # return weights for different models and take logs
        weights = self.evaluate_weights(
            local_parameters=local_parameters,
            number_samples=1
        )
        log_weights = np.log(weights + eps)

        # calculate log likelihoods
        if model_parameters is None:
            log_likelis = np.array(
                [
                    model.log_likelihood_elementwise(y_exp, y_err) + log_weight
                    for model, log_weight in zip(
                        self.models.values(), log_weights
                    )
                ]
            )
        else:
            log_likelis = np.array(
                [
                    model.log_likelihood_elementwise(y_exp, y_err, *parameters)
                    + log_weight
                    for model, parameters, log_weight in zip(
                        self.models.values(),
                        model_parameters.values(),
                        log_weights,
                    )
                ]
            )

        mix_log_likeli = np.logaddexp.reduce(log_likelis)
        return mix_log_likeli.item()

    ##########################################################################
    ##########################################################################

    def plot_weights(self, local_parameters: np.ndarray) -> np.ndarray:
        """
        plot the mixing function against the input parameter values x

        Parameters
        ----------
        local_parameters : np.1darray
            parameters to determine where to sample the prior distirbution

        Returns:
        --------
        None
        """
        # TODO: Does this function make sense here, how to plot meaninguflly
        weights = self.evaluate_weights(local_parameters)
        fig, ax = plt.subplots()
        ax.scatter(np.arange(len(weights)), weights)
        ax.set_xlabel("Model number")
        ax.set_ylabel("Model weight")
        return None

    ##########################################################################
    ##########################################################################

    @property
    def posterior(self):
        """
        Stores the most recent posteriors from running self.train function

        Returns:
        --------
        self.m_posterior : np.ndarray
            posterior from learning the weights
        """
        if self.has_trained:
            return self.m_posterior
        else:
            raise Exception("Please train model before requesting posterior")

    ##########################################################################
    ##########################################################################

    def _sample_distribution(
        self,
        distribution: np.ndarray,
        model_parameters: Optional[Dict[str, List[float]]] = None,
    ) -> np.ndarray:
        """
        Helper function to evaluate the predictive distribution from a given
        posterior distribution

        Parameters:
        -----------
        distribution : np.ndarray
            Can be the MCMC chain from the posterior sampling, or a set of
            sample points selected by the user
        model_parameters : Optional[Dict[str, List[float]]]
            dictionary contain lists of the parameters each model needs to use

        Returns:
        --------
        sample : np.ndarray
            array of points where mixing model was evaluated at
        """
        return np.array(
            [
                self.evaluate(sample, model_parameters)
                for sample in distribution.reshape(-1, self.n_mix)
            ]
        )

    ##########################################################################
    ##########################################################################

    def predict(
        self,
        local_parameters: np.ndarray,
        model_parameters: Optional[Dict[str, List[float]]] = None,
        credible_intervals=[5, 95],
        samples=None,
    ):
        """
        Evaluate posterior to make prediction at test points.

        Parameters:
        -----------
        local_parameter : np.ndarray
            parameters that determine where to sample the pior distribution
        model_parameters : Optional[Dict[str, List[float]]]
            dictionary contain lists of the parameters each model needs to use
        credible_intervals : Optional[List[float], List[List[float]]]
            list of even number of integers the express which percentiles of
            the posterior distribution to compute and return
        sample : np.ndarray
            User supplied distribution, to use instead of the existing
            distribution

        Returns:
        --------
        evaluated_posterior : np.ndarray
            array of posterior predictive distribution evaluated at provided
            test points
        credible_intervals : np.ndarray
            intervals corresponding for 60%, 90% credible intervals
        mean : np.ndarray
            average mixed model value at each provided test points
        std_dev : np.ndarray
            sample standard deviation of mixed model output at provided test
            points
        """

        if self.has_trained and samples is None:
            predictive_distribution = self._sample_distribution(
                self.m_posterior, model_parameters
            )

            return_intervals = np.percentile(
                predictive_distribution, np.asarray(credible_intervals), axis=0
            )
            return_mean = np.mean(predictive_distribution)
            return_stddev = np.std(predictive_distribution)
            return (
                predictive_distribution,
                return_intervals,
                return_mean,
                return_stddev,
            )
        else:
            if samples is None:
                raise Exception(
                    "Please either train model, or provide samples as an argument"
                )
            else:
                predictive_distribution = self._sample_distribution(
                    samples, model_parameters
                )

                return_intervals = np.percentile(
                    predictive_distribution, credible_intervals
                )
                return_mean = np.mean(predictive_distribution)
                return_stddev = np.std(predictive_distribution)
                return (
                    predictive_distribution,
                    return_intervals,
                    return_mean,
                    return_stddev,
                )

            raise Exception("Please train model before making predictions")

    ##########################################################################
    ##########################################################################

    @property
    def prior(self):
        """
        Dictionary of prior distributions. Format should be compatible with
        sampler.

        Returns:
        --------
        self.m_prior : Dict[str, Any]
            Underlying prior object(s)
        """
        if self.m_prior is not None:
            return self.m_prior
        else:
            print("No priors have been set")
            return None

    ##########################################################################
    ##########################################################################

    def prior_predict(
        self,
        loca_parameters: np.ndarray,
        model_parameters: Optional[Dict[str, List[float]]] = None,
        credible_interval=[5, 95],
        number_samples: int = 1000,
    ) -> np.ndarray:
        """
        Get prior predictive distribution and prior distribution samples

        Parameters:
        -----------
        local_parameters : np.ndarray
            parameters that determine where to sample prior distribution
        model_parameters : Optional[Dict[str, List[float]]]
            dictionary contain lists of the parameters each model needs to use
        credible_intervals : Optional[List[float], List[List[float]]]
            list of even number of integers the express which percentiles of
            the posterior distribution to compute and return
        sample : np.ndarray
            User supplied distribution, to use instead of the existing
            distribution

        Returns:
        --------
        evaluated_prior : np.ndarray
            array of prior predictive distribution evaluated at provided
            test points
        credible_intervals : np.ndarray
            intervals corresponding for 60%, 90% credible intervals
        mean : np.ndarray
            average mixed model value at each provided test points
        std_dev : np.ndarray
            sample standard deviation of mixed model output at provided test
            points
        """

        prior_points = self._sample_prior(
            local_parameters=loca_parameters,
            number_samples=number_samples
        )
        prior_points = np.exp(prior_points)
        prior_points = np.array(
            [dirichlet(prior_point).rvs() for prior_point in prior_points]
        )
        return self.predict(
            model_parameters=model_parameters,
            credible_interval=credible_interval,
            samples=prior_points,
        )

    ##########################################################################
    ##########################################################################

    def predict_weights(
        self,
        local_parameters: np.ndarray,
        credible_interval=[5, 95],
        existing_samples=None
    ) -> np.ndarray:
        """
        Calculate posterior predictive distribution for model weights

        Parameters:
        -----------
        local_parameters : np.ndarray
            parameters that determine where to sample the prior distribution
        credible_intervals : Optional[List[float], List[List[float]]]
            list of even number of integers the express which percentiles of
            the posterior distribution to compute and return
        sample : np.ndarray
            User supplied distribution, to use instead of the existing
            distribution

        Returns:
        --------
        evaluated_posterior : np.ndarray
            array of posterior predictive distribution evaluated at provided
            test points
        mean : np.ndarray
            average mixed model value at each provided test points
        credible_intervals : np.ndarray
            intervals corresponding for 60%, 90% credible intervals
        std_dev : np.ndarray
            sample standard deviation of mixed model output at provided test
            points
        """

        # FIXME: Currently, I the self.evaluate_weights function expects to 
        #        take the variables that determines the log-link functions.
        #        However, we would like for it to taje  the sampled parameters
        #        of the Guassian distribution.
        #        A way to do this is have the priors set on the mu, ell and 
        #        sigma parameters and create a new GPR every sampling step.
        #        This seems like an awfully inefficient algorithm though.

        if self.has_trained:
            if existing_samples is None:
                return np.array(
                    [
                        self.evaluate_weights(
                                local_parameters=local_parameters
                        )
                        for _ in self.m_posterior.reshape(-1, self.n_mix)
                    ]
                )
            else:
                # TODO: check that existing_samples has the right shape
                return np.array(
                    [
                        self.evaluate_weights(sample)
                        for sample in existing_samples
                    ]
                )
        else:
            raise Exception("Please train model before making predictions")

    ##########################################################################
    ##########################################################################

    def _sample_prior(
            self,
            local_parameters: np.ndarray,
            number_samples: int
    ) -> np.ndarray:
        """
        Helper function to sample prior since all prior distributions are
        stochastic in nature

        Parameters:
        -----------
        local_parmeters : np.ndarray
            parameters that help determine where to sample prior
        number_samples : int
            number of samples form prior distributions

        Returns:
        --------
        samples : np.ndarray
            array of samples with the shape (number_samples, self.n_mix)
        """

        # TODO: Implement local_parameters
        #       (2) Where we use a linear regression model to learn weights
        prior_samples = []
        for n in np.arange(number_samples):
            samples = []
            for key, priors in self.m_prior:
                random_length = np.random.uniform(*priors[0])
                random_variance = np.random.uniform(*priors[1])
                gpr.kernel_.set_params(
                    {
                        'k1': random_variance ** 2,
                        'k2': RBF(length_scale=random_length),
                    }
                )
                samples.append(gpr.predict(local_parameters))
            prior_samples.append(samples)

        prior_samples = np.vstack(prior_samples)
        if number_samples == 1:
            return prior_samples[0]
        else:
            return prior_samples

    ##########################################################################
    ##########################################################################

    def set_prior(
            self,
            length_scale_bounds: np.ndarray,
            variance_bounds: np.ndarray = np.array([-5, 5]),
            noise_bounds: np.ndarray = 0
    ):
        """
        The prior distribution for log link functions (i.e. the log of the
        shape parameters for the Dirichlet distribution are always set to
        Gaussian Process with radial basis functions as kernels

        Parameters:
        -----------
        legnth_scale_bounds : np.ndarray
            Should have shape (1, 2).
            Range of values for length scale parameters of the Gaussian
            processes: used as uniform prior in length_scale.
        variance_bounds : np.ndarray
            Should have shape (1, 2)
            Range of values for variance parameters of the Gaussian process:
            used as uniform prior on variance
        noise : np.ndarray
            Since kernels for Gaussian process can hav white noise generators,
            the noise parameter is there to set the noise level.
            Can have shape (m,) where m is the number of local variables used
            to caculate the shape parameters.
            (Note, this feature is currently not available)
        """
        # TODO: Implement priors, note that in general the prior distributions
        #       of the hyperparameters of the Dirichlet distribution should
        #       follow a log normal
        #       The type of prior distributions determines the values for the
        #       local parameters.

        kernel = np.squeeze(np.diff(variance_bounds, axis=1)) / 2 \
            * RBF(length_scale=np.squeeze(
                          np.diff(length_scale_bounds, axis=1)
                      ) / 2,
                  length_scale_bounds=length_scale_bounds)
        self.m_gpr = gpr(kernel=kernel)

        prior_dict = {
            f"param_{i}":  [length_scale_bounds, variance_bounds]
            for i in range(self.m_mix_)
        }
        self.m_prior = prior_dict

    ##########################################################################
    ##########################################################################

    def _loglikelihood_for_sampler(
        self,
        local_parameters: np.ndarray,
        y_exp: np.ndarray,
        y_err: np.ndarray,
        model_parameters: Optional[Dict[str, List[float]]] = None,
    ) -> np.ndarray:
        """
        Helper for ptemcee call fo loglikelihood function
        Simply rearranges the order of the loglikelihood function

        Parameters
        ----------
        mix_parameters : np.1darray
            parameter values that fix the shape of mixing function
        y_exp : np.ndarray
            The experimental data
        y_err : np.ndarray
            Gaussian error bars on the experimental data
        model_parameters : Optional[Dict[str, List[float]]]
            list of model parameters for each model, note that different models
            can take different number of parameters

        Returns:
        --------
        log_likelihood : float
            log_likelihood of the model given map parameters for the models and
            a set of weights
        """
        return self.mix_loglikelihood(
            y_exp=y_exp,
            y_err=y_err,
            local_parameters=local_parameters,
            model_parameters=model_parameters,
        )

    ##########################################################################
    ##########################################################################

    def _log_prior(self, prior_parameters: np.ndarray) -> np.ndarray:
        """
        Helper for ptemcee call fo log-prior function
        Simply rearranges the order of the log-prior function

        Parameters:
        -----------
        prior_parameteres : np.ndarray
            parameters to pass to prior distribution and sample it

        Returns:
        --------
        log_prior : float
            The prior distriution evaluated at the provide prior_parameters
            point(s)
        """
        # TODO: Needs to conform with the chosen prior form.
        log_priors = []
        for priors in self.m_prior.values():
            for n, prior in enumerate(priors):
                if (prior_parameters[n] < prior[1]
                        and prior_parameters[n] > prior[0]):
                    log_priors.append(-np.log(np.diff(prior)))
                else:
                    log_priors.append(-np.inf)
        return np.sum(log_priors)

    ##########################################################################
    ##########################################################################

    def train(
        self,
        y_exp: np.ndarray,
        y_err: np.ndarray,
        initial_local_parameters: np.ndarray,
        model_parameters: Optional[Dict[str, List[float]]] = None,
        steps: int = 2000,
        burn: int = 50,
        temps: int = 10,
        walkers: int = 20,
        thinning: int = 100,
    ):
        """
        Run sampler to learn weights. Method should also create class
        members that store the posterior and other diagnostic quantities
        import for plotting
        MAP values should also caluclate and set as member variable of
        class

        Parameters:
        -----------
        y_exp : np.ndarray
            experimental observables to compare models with
        y_err : np.ndarray
            gaussian error bars on observables
        initial_local_parameters : np.ndarray
            parameters that determine where to sample the prior distribution,
            for initializing sampler
        model_parameters : Optional[Dict[str, List[float]]]
            dictionary which contains list of model parameters for each model
        steps: int
            Number of steps for MCMC per model (defaults to 2000)
        burn: int
            Number of burn-in steps for MCMC per model (defaults to 50)
        temps: int
            Number of temperatures to use for parallel termpering (defaults
            to 10)
        walkers: int
            Number of walkers per model (defaults to 20)
        thinning: int
            Keep every `thinning`-th step from MCMC run

        Return:
        -------
        self.m_posterior : np.ndarray
            the mcmc chain return from sampler
        """
        import ptemcee

        nsteps = steps * self.n_mix
        nburn = burn * self.n_mix
        ntemps = temps
        nwalkers = walkers * self.n_mix

        starting_guess = np.array(
            [
                self._sample_prior(
                    local_parameters=initial_local_parameters,
                    number_samples=nwalkers
                )
                for _ in range(ntemps)
            ]
        )
        sampler = ptemcee.Sampler(
            nwalkers=nwalkers,
            dim=self.n_mix,
            ntemps=ntemps,
            Tmax=10,
            threads=8,
            logl=self._loglikelihood_for_sampler,
            logp=self._log_prior,
            loglargs=[y_exp, y_err, model_parameters],
            logpargs=[],
        )
        print("Burn-in sampling")
        x = sampler.run_mcmc(
            p0=starting_guess, iterations=nburn, swap_ratios=True
        )
        print("Burn-in samping complete")
        sampler.reset()
        print("Now running other samples")
        x = sampler.run_mcmc(
            p0=x[0],
            iterations=nsteps,
            thin=thinning,
            storechain=True,
            swap_ratios=True
        )

        # We want the zero temperature chain
        # Recall that the shape of the chain will be:
        #   (ntemps, nwalkers, nsteps, nvars)
        self.m_posterior = np.array(sampler.chain[0, ...])
        self.evidence = sampler.log_evidence_estimate()

        del sampler
        self.m_map = self.m_posterior[np.argmax(self.m_posterior, axis=2)]

        self.has_trained = True
        return self.m_posterior


##########################################################################
##########################################################################
##########################################################################
