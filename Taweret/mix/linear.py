# This will have all the linear bayesian model mixing methods.
# Takes Models as inputs:
# Check if Models have an predict method and they should output a mean and a
# variance.
#
# Modified by K. Ingles

import _mypackage

from Taweret.core.base_mixer import BaseMixer
from Taweret.core.base_model import BaseModel

import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import cpu_count
from scipy.stats import norm, dirichlet
from typing import Any, Dict, Optional, Type, List, Callable
from pathlib import Path

# from sklearn.gaussian_process import GaussianProcessRegressor as gpr
# from sklearn.gaussian_process.kernels import RBF

from bilby.core import prior as bilby_prior
from bilby import Likelihood as Bilby_Likelihood
from bilby import run_sampler as bilby_run_sampler

eps = 1.0e-20


# TODO: Class currently incapable of running simultaneous calibration
# TODO: Update `predict_weigts` and `plot_weight` functions for both below
class LinearMixerGlobal(BaseMixer):
    """
    Generates a global linear mixed model

    """

    def __init__(
            self,
            models: Dict[str, Type[BaseModel]],
            nargs_for_each_model: Optional[np.ndarray] = None,
            n_mix: int = 0
    ):
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

    def evaluate(
            self,
            shape_parameters: np.ndarray,
            model_parameters: Optional[Dict[str, List[Any]]] = None,
    ) -> np.ndarray:
        """
        evaluate mixed model for given mixing function parameters

        Parameters
        ----------
        mixture_parameters : np.ndarray
            parameter values that fix the shape of mixing function
        model_parameters : Optional[Dict[str, List[Any]]]
            Dictionary with list of parameter values for each model


        Returns
        -------
        evaluation : float
            evaluation of the mixing model
        """
        weights = self.evaluate_weights(shape_parameters)

        if self.nargs_for_each_model is None:
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
            shape_parameters: np.ndarray
    ) -> np.ndarray:
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
        return np.array([dirichlet(params) for params in shape_parameters])

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

    # ------------------------------------------------------------------------
    class MixLikelihood(Bilby_Likelihood):
        '''
        Helper class need to run bilby sampler. Wraps around `mix_likelihood`
        function

        '''

        def __init__(
                self,
                keys: List[str],
                likelihood_func: Callable,
                y_exp: np.ndarray,
                y_err: np.ndarray,
                model_parameters: Optional[Dict[str, List[Any]]] = None
        ):
            parameter_dictionary = dict(
                (key, None)
                for key in keys
            )
            super().__init__(
                parameters=parameter_dictionary
            )
            self.likelihood_func = likelihood_func
            self.y_exp = y_exp
            self.y_err = y_err
            self.model_parameters = model_parameters

        def log_likelihood(self):
            samples = np.array(list(self.parameters.values()))
            samples = np.transpose(samples)
            weights = dirichlet(samples).rvs()[0]
            return self.likelihood_func(
                y_exp=self.y_exp,
                y_err=self.y_err,
                weights=weights,
                model_parameters=self.model_parameters
            )
    # ------------------------------------------------------------------------

    def mix_loglikelihood(
            self,
            y_exp: np.ndarray,
            y_err: np.ndarray,
            weights: np.ndarray,
            model_parameters: Optional[Dict[str, List[Any]]] = None,
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
        model_parameters : Optional[Dict[str, List[Any]]]
            list of model parameters for each model, note that different models
            can take different number of parameters

        Returns:
        --------
        log_likelihood : float
            log_likelihood of the model given map parameters for the models and
            a set of weights
        """

        log_weights = np.log(weights + eps)

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

    def plot_weights(
            self,
    ) -> np.ndarray:
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
        weights = self._sample_prior(number_samples=1)
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
            model_parameters: Optional[Dict[str, List[Any]]] = None,
    ) -> np.ndarray:
        """
        Helper function to evaluate the predictive distribution from a given
        posterior distribution

        Parameters:
        -----------
        distribution : np.ndarray
            Can be the MCMC chain from the posterior sampling, or a set of
            sample points selected by the user
        model_parameters : Optional[Dict[str, List[Any]]]
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
            model_parameters: Optional[Dict[str, List[Any]]] = None,
            credible_intervals: List[float] = [5, 95],
            samples: Optional[np.ndarray] = None,
    ):
        """
        Evaluate posterior to make prediction at test points.

        Parameters:
        -----------
        model_parameters : Optional[Dict[str, List[Any]]]
            dictionary contain lists of the parameters each model needs to use
        credible_intervals : Optional[np.ndarray, List[np.ndarray]]
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
                distribution=self.m_posterior,
                model_parameters=model_parameters
            )

            return_intervals = np.percentile(
                a=predictive_distribution,
                q=np.asarray(credible_intervals),
                axis=0
            )
            return_mean = np.mean(predictive_distribution, axis=0)
            return_stddev = np.std(predictive_distribution, axis=0)
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
            model_parameters: Optional[Dict[str, List[Any]]] = None,
            credible_interval=[5, 95],
            number_samples: int = 1000,
    ) -> np.ndarray:
        """
        Get prior predictive distribution and prior distribution samples

        Parameters:
        -----------
        model_parameters : Optional[Dict[str, List[Any]]]
            dictionary contain lists of the parameters each model needs to use
        credible_intervals : Optional[np.ndarray, List[np.ndarray]]
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

        predictive_distribution = self.evaluate(
            weights=self.evaluate_weights(
                self._sample_prior(number_samples=number_samples)
            ),
            model_parameters=model_parameters
        )

        return_intervals = np.percentile(
            a=predictive_distribution,
            q=np.asarray(credible_interval),
            axis=0
        )
        return_mean = np.mean(predictive_distribution, axis=0)
        return_stddev = np.std(predictive_distribution, axis=0)
        return (
            predictive_distribution,
            return_intervals,
            return_mean,
            return_stddev,
        )

    ##########################################################################
    ##########################################################################

    def predict_weights(
            self,
            credible_interval: List[float] = [5, 95],
            samples: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate posterior predictive distribution for model weights

        Parameters:
        -----------
        credible_intervals : Optional[np.ndarray, List[np.ndarray]]
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
                    dirichlet(sample).rvs()[0]
                    for sample in self.m_posterior.reshape(-1, self.n_mix)
                ]
            )
        else:
            raise Exception("Please train model before making predictions")

    ##########################################################################
    ##########################################################################

    def _sample_prior(
            self,
            number_samples: int
    ) -> np.ndarray:
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

        # Bilby returns dictionary with where is values is an array of length
        # number_samples, we then append the last weight using the simplex
        # condition
        samples = self.m_prior.sample(size=number_samples)
        samples = np.array(list(samples.values()))
        samples = np.transpose(samples)
        return samples

    ##########################################################################
    ##########################################################################

    def set_prior(
            self,
    ):
        """
        A call to this function automatically sets up a dictionary of length
        self.n_mix. Each weight is labelled by `label`+ f'_{num}'

        Parameters:
        -----------
        label : Optional[str]
            label with which to identify weights for mixing likelihood. Used
            internally by bilby
        """
        print("Notice: For global fitting, the prior is always assumed to be")
        print("        a Dirichlet distribution")

        # In bilby, if I `n_dim = N`, then it only stores `N - 1` weights
        # since the last weight is determined by the simplex conditons
        #
        # Perhaps what we want to do is to store log link functions and pass
        # the samples to a dirichlet function
        self.m_prior = bilby_prior.PriorDict(
            dict(
                (f'alpha_{n}', bilby_prior.LogNormal(0, 1, f'alpha_{n}'))
                for n in range(self.n_mix)
            )
        )

    ##########################################################################
    ##########################################################################

    def train(
            self,
            y_exp: np.ndarray,
            y_err: np.ndarray,
            outdir: Path,
            model_parameters: Optional[Dict[str, List[Any]]] = None,
            kwargs_for_sampler: Optional[Dict[str, Any]] = None,
            steps: int = 2000,
            burn: int = 50,
            temps: int = 10,
            walkers: int = 20,
            # thinning: int = 100,
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
        model_parameters : Optional[Dict[str, List[Any]]]
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

        if kwargs_for_sampler is None:
            kwargs_for_sampler = {
                'sampler': 'ptemcee',
                'ntemps': temps,
                'Tmax': 100,
                'nwalkers': walkers * self.n_mix,
                'nsamples': steps * self.n_mix,
                'burn_in_fixed_discard': burn * self.n_mix,
                'threads': cpu_count(),
                'clean': True,
                'printdt': 60
            }

        result = bilby_run_sampler(
            likelihood=self.MixLikelihood(
                keys=list(self.m_prior.keys()),
                likelihood_func=self.mix_loglikelihood,
                y_exp=y_exp,
                y_err=y_err,
                model_parameters=model_parameters
            ),
            priors=self.m_prior,
            outdir=str(outdir),
            **kwargs_for_sampler
        )
        result.plot_corner()

        self.m_posterior = np.array(
            [
                result.posterior[var] for var in self.m_prior.keys()
            ]
        )
        self.m_posterior = np.transpose(self.m_posterior)

        self.evidence = result.log_10_evidence

        hists = [
            np.histogram(
                a=samples,
                bins=int(np.floor(np.sqrt(samples.size)))
            )
            for samples in np.transpose(self.m_posterior)
        ]
        self.m_map = np.array(
            [
                bins[np.argmax(hist)] + np.diff(bins) / 2
                for hist, bins in hists
            ]
        )

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

    def evaluate(
        self,
        local_variables: np.ndarray,
        model_parameters: Optional[Dict[str, List[Any]]] = None,
        sample: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        evaluate mixed model for given mixing function parameters

        Parameters
        ----------
        local_variables : np.ndarray
            parameters that determine where to sample the prior distributions
        model_parameters : Optional[Dict[str, List[Any]]]
            Dictionary with list of parameter values for each model
        sample : Optional[Dict[str, np.ndarray]]
            optional argument, useful for when the prior has already been
            sampled, such as when running MCMC. It is automatically assumed
            that sample is of shape (2 * self.n_local_variables,)


        Returns
        -------
        evaluation : float
            evaluation of the mixing model
        """

        # FIXME: I am currently returning the weights, but I should be
        #        sampling the log-normal distribution for the dirichlet
        #        hyperparameters
        weights = self.evaluate_weights(
            local_variables=local_variables,
            number_samples=1,
            sample=sample
        )

        if self.nargs_for_each_model is None:
            return np.sum(
                [
                    weight * model.evaluate(np.squeeze(local_variables))
                    for weight, model in zip(weights, self.models.values())
                ],
                axis=0
            )
        else:
            return np.sum(
                [
                    weight * model.evaluate(np.squeeze(local_variables),
                                            *parameters)
                    for weight, model, parameters in zip(
                        weights,
                        self.models.values(),
                        model_parameters.values(),
                    )
                ],
                axis=0
            )

    ##########################################################################
    ##########################################################################

    def evaluate_weights(
        self,
        local_variables: np.ndarray,
        number_samples: int = 1,
        sample: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        calculate the weights given some set of input parameters

        Parameters
        ----------
        local_variables : np.ndarray
            should have shape (self.n_local_variables,)
            local dependence of weights, such as centrality for heavy-ion
            collisions
        number_samples : int
            number of samples to return from prior
        sample : Optional[Dict[str, np.ndarray]]
            optional argument, useful for when the prior has already been
            sampled, such as when running MCMC. It is automatically assumed
            that sample is of shape (2 * self.n_local_variables,)

        Returns:
        --------
        weights : np.ndarray
            array of sampled weights
        """
        if self.deterministic:
            prior_samples = self._sample_prior(number_samples=number_samples) \
                            if sample is None else sample
            prior_samples = np.array(list(prior_samples.values()))
            prior_samples = prior_samples.reshape(
                2 * self.n_mix * self.n_local_variables, -1
            )
            weights = np.zeros((self.n_mix, number_samples))
            for n in range(self.n_mix - 1):
                for i in range(number_samples):
                    weights[n, i] = np.prod(np.array(
                        [
                            np.exp(
                                prior_samples[2 * k + 0, i] *
                                local_variables[k] +
                                prior_samples[2 * k + 1, i]
                            )
                            if self.polynomial_order == 1 else
                            np.exp((local_variables[k]
                                    - prior_samples[2 * k + 0, i]) ** 2
                                   + prior_samples[2 * k + 1, i])
                            # np.exp(
                            #     norm(
                            #         loc=prior_samples[2 * k + 0, i],
                            #         scale=prior_samples[2 * k + 1, i]
                            #     ).pdf(local_variables[k])
                            # )
                            for k in range(self.n_local_variables)
                        ])
                    )
            for n in range(self.n_mix - 1):
                weights[n] = weights[n] / (1 + np.sum(weights, axis=0))
            for i in range(number_samples):
                weights[self.n_mix - 1, i] = 1 - np.sum(weights[:, i])
            return np.squeeze(weights)

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

    # ------------------------------------------------------------------------
    class MixLikelihood(Bilby_Likelihood):
        '''
        Helper class need to run bilby sampler. Wraps around `mix_likelihood`
        function

        '''

        def __init__(
                self,
                keys: List[str],
                likelihood_func: Callable,
                evaluate_weights: Callable,
                y_exp: np.ndarray,
                y_err: np.ndarray,
                local_variables: np.ndarray,
                model_parameters: Optional[Dict[str, List[Any]]] = None
        ):
            parameter_dictionary = dict(
                (key, None)
                for key in keys
            )
            super().__init__(
                parameters=parameter_dictionary
            )
            self.likelihood_func = likelihood_func
            self.evaluate_weights = evaluate_weights
            self.y_exp = y_exp
            self.y_err = y_err
            self.local_variables = local_variables
            self.model_parameters = model_parameters

        def log_likelihood(self):
            weights = self.evaluate_weights(
                local_variables=self.local_variables,
                sample=self.parameters
            )
            return self.likelihood_func(
                y_exp=self.y_exp,
                y_err=self.y_err,
                weights=weights,
                local_variables=self.local_variables,
                model_parameters=self.model_parameters
            )
    # ------------------------------------------------------------------------

    def mix_loglikelihood(
            self,
            y_exp: np.ndarray,
            y_err: np.ndarray,
            weights: np.ndarray,
            local_variables: np.ndarray,
            model_parameters: Optional[Dict[str, List[Any]]] = None,
    ) -> float:
        """
        log likelihood of the mixing model

        Parameters
        ----------
        y_exp : np.ndarray
            The experimental data
        y_err : np.ndarray
            Gaussian error bars on the experimental data
        local_variables : np.ndarray
            local dependence of weights, e.g. centrality in heavy-ion colliions
        model_parameters : Optional[Dict[str, List[Any]]]
            list of model parameters for each model, note that different models
            can take different number of parameters

        Returns:
        --------
        log_likelihood : float
            log_likelihood of the model given map parameters for the models and
            a set of weights
        """

        # FIXME: The model should not need to consume the local_variables
        #        Perhaps an option that allows for the specifying of positional
        #        parameters?

        # In general, the weights statisfy a simplex condition (they sum to 1)
        # A natural distribution to choose for the weights is a dirichlet
        # distribution, which hyperparameters that need priors specified
        log_weights = np.log(weights + eps)

        # calculate log likelihoods
        if model_parameters is None:
            log_likelis = np.array(
                [
                    model.log_likelihood_elementwise(
                            y_exp,
                            y_err,
                            local_variables
                        ) + log_weight
                    for model, log_weight in zip(
                        self.models.values(), log_weights
                    )
                ]
            )
        else:
            log_likelis = np.array(
                [
                    model.log_likelihood_elementwise(
                            y_exp,
                            y_err,
                            local_variables,
                            *parameters
                        ) + log_weight
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

    def plot_weights(self, local_variables: np.ndarray) -> np.ndarray:
        """
        plot the mixing function against the input parameter values x

        Parameters
        ----------
        local_variables : np.1darray
            parameters to determine where to sample the prior distirbution

        Returns:
        --------
        None
        """
        # TODO: Does this function make sense here, how to plot meaninguflly
        weights = self.evaluate_weights(local_variables)
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
            local_variables: np.ndarray,
            model_parameters: Optional[Dict[str, List[Any]]] = None,
    ) -> np.ndarray:
        """
        Helper function to evaluate the predictive distribution from a given
        posterior distribution

        Parameters:
        -----------
        distribution : np.ndarray
            Can be the MCMC chain from the posterior sampling, or a set of
            sample points selected by the user
        model_parameters : Optional[Dict[str, List[Any]]]
            dictionary contain lists of the parameters each model needs to use

        Returns:
        --------
        sample : np.ndarray
            array of points where mixing model was evaluated at
        """
        return np.array(
            [
                self.evaluate(
                    local_variables=local_variables,
                    model_parameters=model_parameters,
                    sample=dict(
                        (key, var) for key, var in zip(self.m_prior.keys(),
                                                       sample)
                    )
                )
                for sample in distribution  # we assume it has correct shape
            ]
        )

    ##########################################################################
    ##########################################################################

    def predict(
            self,
            local_variables: np.ndarray,
            model_parameters: Optional[Dict[str, List[Any]]] = None,
            credible_intervals=[5, 95],
            samples=None,
    ):
        """
        Evaluate posterior to make prediction at test points.

        Parameters:
        -----------
        local_parameter : np.ndarray
            parameters that determine where to sample the pior distribution
        model_parameters : Optional[Dict[str, List[Any]]]
            dictionary contain lists of the parameters each model needs to use
        credible_intervals : Optional[np.ndarray, List[np.ndarray]]
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
            predictive_distribution = np.array(
                [
                    self._sample_distribution(
                        distribution=self.m_posterior,
                        local_variables=x,
                        model_parameters=model_parameters
                    )
                    for x in local_variables.reshape(-1,
                                                     self.n_local_variables)
                ]
            )

            return_intervals = np.percentile(
                predictive_distribution, np.asarray(credible_intervals), axis=1
            )
            return_mean = np.mean(predictive_distribution, axis=1)
            return_stddev = np.std(predictive_distribution, axis=1)
            return (
                predictive_distribution,
                return_intervals,
                return_mean,
                return_stddev,
            )
        else:
            if samples is None:
                raise Exception(
                    "Please either train model, or provide samples as an"
                    + "argument"
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
            local_variables: np.ndarray,
            model_parameters: Optional[Dict[str, List[Any]]] = None,
            credible_interval=[5, 95],
            number_samples: int = 1000,
    ) -> np.ndarray:
        """
        Get prior predictive distribution and prior distribution samples

        Parameters:
        -----------
        local_variables : np.ndarray
            parameters that determine where to sample prior distribution
        model_parameters : Optional[Dict[str, List[Any]]]
            dictionary contain lists of the parameters each model needs to use
        credible_intervals : Optional[np.ndarray, List[np.ndarray]]
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

        # FIXME: This function is still form global inference
        prior_points = self._sample_prior(
            local_variables=local_variables,
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
            local_variables: np.ndarray,
            credible_interval=[5, 95],
            existing_samples=None
    ) -> np.ndarray:
        """
        Calculate posterior predictive distribution for model weights

        Parameters:
        -----------
        local_variables : np.ndarray
            parameters that determine where to sample the prior distribution
        credible_intervals : Optional[np.ndarray, List[np.ndarray]]
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
                                local_variables=local_variables
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
            local_variables: np.ndarray,
            weight_distribution_hyperparameters: Dict[str, np.ndarray],
            number_samples: int
    ) -> np.ndarray:
        """
        Helper function to sample prior since all prior distributions are
        stochastic in nature

        Parameters:
        -----------
        local_variables : np.ndarray
            parameters that help determine where to sample prior
        number_samples : int
            number of samples form prior distributions

        Returns:
        --------
        samples : np.ndarray
            array of samples with the shape (number_samples, self.n_mix)
        """
        return self.m_prior.sample(size=number_samples)

    ##########################################################################
    ##########################################################################

    def set_prior(
            self,
            example_local_variable: np.ndarray,
            local_variables_ranges: np.ndarray,
            deterministic_priors: bool = False,
            universal_priors: Optional[np.ndarray] = None,
            priors_dicitionary: Optional[
                Dict[str, Type[bilby_prior.Prior]]
            ] = None,
            polynomial_order: Optional[int] = None,
    ):
        """
        The prior distribution for log link functions (i.e. the log of the
        shape parameters for the Dirichlet distribution are always set to
        Gaussian Process with radial basis functions as kernels

        Parameters:
        -----------
        example_local_variables : np.ndarray
            should a be an array with shape(n,) which lets Taweret know
            whether to use a 1-d priors or multi-dimensional priors
        local_variables_ranges : np.ndarray
            should be array with shape (n, 2), determines the parameter
            range for both the (slope, intercept) [for linear] or (h, k) [for
            parabola]
        deterministic_priors : boo
            (default = False)
            determines whether the likelihoods will be deterministic or
            stochastic
        universal_prior : Optional[np.ndarray]
            (default = None)
            Passing this object sets all priors to the same prior
        priors_dictionary : Optional[Dict[str, Type[bilby_prior.Prior]]]
            (default = None)
            take existing dictionary to pass to bilby
        polynomial_order : Optional[int]
            (default = None)
            if using deterministic priors, this determines the order of
            the :math: `W_h(x)` function, see Coleman Thesis pg. 82 
        """

        self.deterministic = deterministic_priors
        if deterministic_priors:

            if polynomial_order is None:
                print('No polynomial order is given, assuming 2 by default')
                self.polynomial_order = 2
            elif polynomial_order > 2:
                print('Only powers supported are 1 and 2, setting to 2')
                self.polynomial_order = 2
            else:
                self.polynomial_order = polynomial_order

            self.n_local_variables = example_local_variable.size
            self.m_prior = dict()
            for n in range(1, self.n_mix + 1):
                for k in range(self.n_local_variables):
                    self.m_prior[f'mu_({n}, {k})'] = bilby_prior.Uniform(
                        minimum=local_variables_ranges[k, 0],
                        maximum=local_variables_ranges[k, 1],
                        name=f'mu_({n}, {k})'
                    )
                    self.m_prior[f'sigma_({n}, {k})'] = bilby_prior.Uniform(
                        minimum=local_variables_ranges[k, 0],
                        maximum=local_variables_ranges[k, 1],
                        name=f'sigma_({n}, {k})'
                    )
            self.m_prior = bilby_prior.PriorDict(dictionary=self.m_prior)
        else:
            print('This option is currently not supported')
            exit(-10)
        # kernel = np.squeeze(np.diff(variance_bounds)) / 2 \
        #     * RBF(length_scale=np.squeeze(
        #                   np.diff(length_scale_bounds)
        #               ) / 2,
        #           length_scale_bounds=length_scale_bounds)
        # self.m_gpr = gpr(kernel=kernel)

        # prior_dict = {
        #     f"param_{i}":  local_parameter_ranges
        #     for i in range(self.n_mix)
        # }
        # self.m_prior = prior_dict

    ##########################################################################
    ##########################################################################

    def train(
            self,
            y_exp: np.ndarray,
            y_err: np.ndarray,
            outdir: Path,
            local_variables: np.ndarray,
            model_parameters: Optional[Dict[str, List[Any]]] = None,
            kwargs_for_sampler: Optional[Dict[str, Any]] = None,
            steps: int = 2000,
            burn: int = 50,
            temps: int = 10,
            walkers: int = 20,
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
        initial_local_variables : np.ndarray
            parameters that determine where to sample the prior distribution,
            for initializing sampler
        model_parameters : Optional[Dict[str, List[Any]]]
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

        if kwargs_for_sampler is None:
            kwargs_for_sampler = {
                'sampler': 'ptemcee',
                'ntemps': temps,
                'Tmax': 100,
                'nwalkers': walkers * self.n_mix,
                'nsamples': steps * self.n_mix,
                'burn_in_fixed_discard': burn * self.n_mix,
                'threads': cpu_count(),
                'clean': True,
                'printdt': 60
            }

        result = bilby_run_sampler(
            likelihood=self.MixLikelihood(
                keys=list(self.m_prior.keys()),
                likelihood_func=self.mix_loglikelihood,
                evaluate_weights=self.evaluate_weights,
                y_exp=y_exp,
                y_err=y_err,
                model_parameters=model_parameters,
                local_variables=local_variables
            ),
            priors=self.m_prior,
            outdir=str(outdir),
            **kwargs_for_sampler
        )
        result.plot_corner()

        self.m_posterior = np.array(
            [
                result.posterior[var] for var in self.m_prior.keys()
            ]
        )
        self.m_posterior = np.transpose(self.m_posterior)

        self.evidence = result.log_10_evidence

        # This is crude a wrong, but a start for now
        # MAP needs to be found through optimization, or using bilby API
        hists = [
            np.histogram(
                a=samples,
                bins=int(np.floor(np.sqrt(samples.size)))
            )
            for samples in np.transpose(self.m_posterior)
        ]
        self.m_map = np.array(
            [
                bins[np.argmax(hist)] + np.diff(bins)[0] / 2
                for hist, bins in hists
            ]
        )
        self.has_trained = True
        return self.m_posterior


##########################################################################
##########################################################################
##########################################################################
