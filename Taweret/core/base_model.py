from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def evaluate(self):
        '''
        Calculate a point estimate for model

        Returns:
        --------
        evaluation : float
            point estimate for model

        Example:
        --------

        ```python
        class MyModel(BaseModel):
            @property
            def model(self):
                return self._model

            @models.setter
            def model(self, the_model):
                self._model = the_model

            def evaluate(self, model_parameters):
                return self._model(model_parameters)
            . . .
        ```
        '''
        return NotImplemented

    @abstractmethod
    def log_likelihood_elementwise(self):
        '''
        Calculate log_likelihood for array of points given, and return with
        array with same shape[0]

        Returns:
        --------
        log_likelis : np.ndarray
            an array of length as shape[0] of the input evaluation points

        Example:
        --------
        ```python
        class MyModel(BaseModel):
            def log_likelihood_elementwise(self, y_exp, y_err, model_params):
                # For global mixing, this should only return the log_likelihood
                y = self.evaluate(model_params)
                return np.sum(np.exp(-(y - y_exp) / (2 * y_err) ** 2) \
                    / np.sqrt(2 * np.pi * y_err ** 2))
        ```
        '''
        return NotImplemented

    @abstractmethod
    def set_prior(self):
        '''
        User must provide function that sets a member varibale called _prior.
        Dictionary of prior distributions. Format should be compatible with
        sampler.

        Example:
        --------
        ```python
        class MyModel(BaseMixer):
            . . .
            def set_prior(self, prior_dict):
                self._prior = prior_dict
            . . .
        ```
        '''
        return NotImplemented
