from abc import ABC, abstractmethod


class BaseMixer(ABC):
    @abstractmethod
    def evaluate(self):
        '''
        Calculate a point estimate for mixing model

        Returns:
        --------
        evaluation : float
            point estimate from mixing model

        Example:
        --------
        Global linear mixing:
        $$ f_\mathrm{mixed} = \sum_{i=1}^N w_i \mathcal M_i(\theta_i) $$

        ```python
        class MyMixer(BaseMixer):
            @property
            def models(self):
                return self._model_list

            @models.setter
            def model(self, model_list):
                self._model_list = model_list

            def evaluate(self, weights, model_parameters):
                value = 0
                for i in range(len(self._model_list)):
                    value += self._model_list[i](x, model_parameters[i]) \
                            * weight[i]
                return value
            . . .
        ```
        '''
        return NotImplemented

    @abstractmethod
    def evaluate_weights(self):
        '''
        Calculate or sample a point estimate for model weights

        Returns:
        --------
        weights : np.ndarray
            array of the evaluations or samples of weights

        Example:
        --------
        Global linear mixing:

        ```python
        import scipy

        class MyMixer(BaseMixer):
            . . .

            def evaluate_weights(self, dirichlet_params):
                return scipy.stats.dirichlet.rvs(dirichlet_params)
            . . .
        ```
        '''
        return NotImplemented

    @property
    @abstractmethod
    def map(self):
        '''
        Stores the MAP values for the posterior distributions and is set
        during the self.train step
        '''
        return NotImplemented

    @abstractmethod
    def predict(self):
        '''
        Evaluate posterior to make prediction at test points.

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

        Example:
        --------
        Global liner mixing:

        ```python
        class MyMixer(BaseMixer):
            . . .
            def predict(self, x_test):
                # work to calculate everything
                . . .
                return posterior, means, credible_intervals, std_dev
        ```
        '''
        return NotImplemented

    @abstractmethod
    def predict_weights(self):
        '''
        Calculate posterior predictive distribution for model weights

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

        Example:
        --------
        Global liner mixing:

        ```python
        class MyMixer(BaseMixer):
            . . .
            def predict_weights(self, x_test):
                # work to calculate everything
                . . .
                return posterior, means, credible_intervals, std_dev
        ```
        '''
        return NotImplemented

    @abstractmethod
    def prior_predict(self):
        '''
        Get prior predictive distribution and prior distribution samples
        Returns:
        --------
        evaluated_prior : np.ndarray
            array of prior predictive distribution evaluated at provided
            test points
        mean : np.ndarray
            average mixed model value at each provided test points
        credible_intervals : np.ndarray
            intervals corresponding for 60%, 90% credible intervals
        std_dev : np.ndarray
            sample standard deviation of mixed model output at provided test
            points

        Example:
        --------
        Global liner mixing:

        ```python
        class MyMixer(BaseMixer):
            . . .
            def prior_predict(self, x_test):
                # work to calculate everything
                . . .
                return prior, means, credible_intervals, std_dev
        ```
        '''
        return NotImplemented

    @property
    @abstractmethod
    def posterior(self):
        '''
        Stores the most recent posteriors from running self.train function

        Returns:
        --------
        _posterior : np.ndarray
            posterior from learning the weights
        '''
        return self.posterior

    @property
    @abstractmethod
    def prior(self):
        '''
        Dictionary of prior distributions. Format should be compatible with
        sampler.

        Returns:
        --------
        _prior : Dict[str, Any]
            Underlying prior object(s)

        Example:
        --------
        Please consult `BaseMixer.set_prior` for an example
        '''
        return self.prior

    @abstractmethod
    def set_prior(self):
        '''
        User must provide function that sets a member varibale called _prior.
        Dictionary of prior distributions. Format should be compatible with
        sampler.

        Example:
        --------
        ```python
        class MyMixer(BaseMixer):
            . . .
            def set_prior(self, prior_dict):
                self._prior = prior_dict
            . . .

        # creating a bilby prior dict
        priors = dict()
        priors['a'] = bilby.core.prior.MultivariateGaussian(mvg, 'a')
        priors['b'] = bilby.core.prior.MultivariateGaussian(mvg, 'b')
        m = MyMixer()
        m.set_prior(prior_dict=priors)
        ```
      '''
        return NotImplemented

    @abstractmethod
    def train(self):
        '''
        Run sampler to learn parameters. Method should also create class
        members that store the posterior and other diagnostic quantities
        import for plotting
        MAP values should also caluclate and set as member variable of
        class

        Return:
        -------
        _posterior : np.ndarray
            the mcmc chain return from sampler
        '''
        return NotImplemented
