# SAMBA methods included here: Multivariate BMM
# Written by: Alexandra Semposki
# Authors of SAMBA: Alexandra Semposki, Dick Furnstahl, and Daniel Phillips

# necessary imports
from Taweret.core.base_mixer import BaseMixer
import numpy as np
import sys
sys.path.append('../Taweret')


class Multivariate(BaseMixer):
    r'''
    The multivariate BMM class originally introduced
    in the BAND SAMBA package. Combines individual
    models using a Gaussian form.

    .. math::

        f_{\dagger}
        = \mathcal{N}
        \left(
            \sum_i \frac{f_i/v_i}{1/v_i}, \sum_i \frac{1}{v_i}
        \right)

    Example:
    --------

    .. code-block:: python

            m = Multivariate(x=np.linspace(), models=dict(), n_models=0)
            m.predict(ci=68)
            m.evaluate_weights()
    '''

    def __init__(self, x, models, n_models=0):
        '''
        Parameters:
        -----------
        x : numpy.linspace
            Input space variable in which mixing is occurring.

        models : dict
            Dict of models with BaseModel methods.

        n_models : int
            Number of free parameters per model.

        Returns:
        --------
        None.
        '''

        # check for predict method in the models
        for i in models.keys():
            try:
                getattr(models[i], 'evaluate')
            except AttributeError:
                print('model {i} does not have an evaluate method')

        # set up the class variables
        self.model_dict = models
        self.x = x
        self.n_models = n_models

        # convert models dict() to list
        self.models = [i for i in self.model_dict.values()]

        # set up weights variable
        self.var_weights = np.zeros(len(self.models))

        return None

    def evaluate(self):
        '''
        Evaluate the mixed model at one set of parameters.
        Not needed for this mixing method.
        '''
        return None

    def evaluate_weights(self):
        '''
        Calculate the weights for each model in the mixed model
        over the input space.

        Returns:
        --------
        weights : numpy.ndarray
            Array of model weights calculated in the
            Multivariate.predict function.
        '''

        # check predict() has been called
        if self.var_weights is np.zeros(len(self.models)):
            raise Exception('Please run the predict function before\
                calling this function.')

        # return the weights calculated in the predict method
        return self.var_weights

    @property
    def map(self):
        '''
        Return the MAP values of the parameters.
        Not needed for this method.
        '''
        return None

    @property
    def posterior(self):
        '''
        Return the posterior of the parameters.
        Not needed for this mixing method.
        '''
        return None

    def predict(self, ci=68):
        '''
        The f_dagger function responsible for mixing the models together
        in a Gaussian way. Based on the first two moments of the
        distribution: mean and variance.

        Parameters:
        -----------
        ci : int, list
            The desired credibility interval(s) (1-sigma, 2-sigma)

        Returns:
        --------
        mean, intervals, std_dev : numpy.ndarray
            The mean, credible intervals, and std_dev
            of the predicted mixed model
        '''

        # credibility interval(s)
        self.ci = ci

        # predict for the two models
        self.prediction = []

        for i in range(len(self.models)):
            self.prediction.append(self.models[i].evaluate(self.x))

        # calculate the models from the class variables
        f = []
        v = []

        for i in range(len(self.models)):
            f.append(self.prediction[i][0].flatten())
            v.append(np.square(self.prediction[i][1]).flatten())

        # initialise arrays
        num = np.zeros(len(self.x))
        denom = np.zeros(len(self.x))
        var = np.zeros(len(self.x))

        # sum over the models in the same input space
        for i in range(len(self.x)):
            num[i] = np.sum([f[j][i] / v[j][i] for j in range(len(f))])
            denom[i] = np.sum([1 / v[j][i] for j in range(len(f))])

        # combine everything via input space tracking
        mean = num / denom
        var = 1 / denom

        # variances for each model
        v = np.asarray(v)
        weights = 1.0 / v
        self.var_weights = weights / np.sum(weights, axis=0)

        # std_dev calculation
        std_dev = np.sqrt(var)

        # credibility interval check
        if self.ci == 68:
            val = [1.0]
        elif self.ci == 95:
            val = [1.96]
        elif self.ci == [68, 95]:
            val = [1.0, 1.96]
        else:
            raise ValueError('Choose 1 and/or 2 sigma band.')

        # calculate interval(s)
        interval_low = []
        interval_high = []

        for i in range(len(val)):
            interval_low.append(mean - val[i] * std_dev)
            interval_high.append(mean + val[i] * std_dev)

        # combine interval(s) into one list to return
        interval = [interval_low, interval_high]

        return 0.0, mean, interval, std_dev

    def predict_weights(self):
        '''
        Predict the weights of the mixed model. Returns
        mean and intervals from the posterior of the
        weights.
        Not needed for this mixing method.
        '''
        return None

    @property
    # @prior.setter
    def prior(self):
        '''
        Return the prior of the parameters in the mixing.
        Not needed for this method.
        '''
        return None

    def prior_predict(self):
        '''
        Find the predicted prior distribution.
        Not needed for this mixing method.
        '''
        return None

    def sample_prior(self):
        '''
        Returns samples from the prior
        distributions for the various weight parameters.
        Not needed for this mixing method.
        '''
        return None

    def set_prior(self):
        '''
        Set the priors on the parameters.
        Not needed for this mixing method.
        '''
        return None

    def train(self):
        '''
        Train the mixed model by optimizing the
        weights.
        Not needed in this mixing method.
        '''
        return None
