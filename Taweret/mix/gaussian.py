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
    models using a Gaussian form. Now able to handle
    correlated models (up to the 3 model case). 
    ### Extend to N models later ###

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

    def predict(self, ci=68, correlated=False, R=0, S=0, T=0):
        '''
        The f_dagger function responsible for mixing the models together
        in a Gaussian way. Based on the first two moments of the
        distribution: mean and variance.

        Parameters:
        -----------
        ci : int, list
            The desired credibility interval(s) (1-sigma, 2-sigma)
            
        correlated : bool
            If the models are correlated, will choose the correlated
            model mixing formula. Default is False.
        
        R : int, float
            The correlation parameter between model 1 and model 2.
        
        S : int, float
            The correlation parameter between model 2 and model 3.
            
        T : int float
            The correlation parameter between model 1 and model 3.

        Returns:
        --------
        mean, intervals, std_dev : numpy.ndarray
            The mean, credible intervals, and std_dev
            of the predicted mixed model
        '''
        
        # set var_weights to zero
        self.var_weights = np.zeros([len(self.models), len(self.x)])

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
            
        # decide whether we use correlated or not
        if correlated is True:
            
            # for 2 models
            if self.n_models == 2:
                R = R
                
                # separate the arrays again
                f_1 = np.asarray(f[0])
                f_2 = np.asarray(f[1])
                                
                var_1 = np.asarray(np.sqrt(v[0]))
                var_2 = np.asarray(np.sqrt(v[1]))

                mean_n = np.zeros([len(self.x)])
                mean = np.zeros([len(self.x)])
                v_plus = np.zeros([len(self.x)])
                var = np.zeros([len(self.x)])
                var_n = np.zeros([len(self.x)])

                # write up the mean and variance for the correlated model now
                mean_n = var_1**2.0 * f_2 + var_2**2.0 * f_1 - (R * var_1 * var_2 * (f_1 + f_2))

                v_plus = var_1**2.0 - 2.0 * R * var_1 * var_2 + var_2**2.0

                mean = mean_n/v_plus
                var_n = (var_1**2.0 * var_2**2.0 * (1.0 - R**2.0))
                
                var = var_n/v_plus
            
            # for 3 models
            if self.n_models == 3:
                R = R
                S = S
                T = T
                
                # separate the arrays again
                f_1 = np.asarray(f[0])
                f_2 = np.asarray(f[1])
                f_3 = np.asarray(f[2])
                                
                var_1 = np.asarray(np.sqrt(v[0]))
                var_2 = np.asarray(np.sqrt(v[1]))
                var_3 = np.asarray(np.sqrt(v[2]))
                
                #initialise arrays
                mean_n = np.zeros([len(self.x)])
                mean = np.zeros([len(self.x)])
                v_plus = np.zeros([len(self.x)])
                var_n = np.zeros([len(self.x)])
                var = np.zeros([len(self.x)])

                # cofactor matrix elements
                v_plus = (var_2**2.0 * var_3**2.0 * (1.0 - S**2.0)) + (var_1**2.0 * var_3**2.0 * (1.0 - T**2.0)) \
                + (var_1**2.0 * var_2**2.0 * (1.0 - R**2.0)) - 2.0*(var_1*var_2*var_3**2.0 * (R - S*T))  \
                + 2.0*(var_1*var_2**2.0*var_3 * (R*S - T)) - 2.0*(var_1**2.0*var_2*var_3 * (S - R*T))

                # mean function
                mean_n = (var_2**2.0 * var_3**2.0 * (1.0 - S**2.0) * f_1) + (var_1**2.0 * var_3**2.0 * (1.0 - T**2.0) * f_2) \
                + (var_1**2.0 * var_2**2.0 * (1.0 - R**2.0) * f_3) - (var_1*var_2*var_3**2.0 * (R - S*T) * (f_1 + f_2))  \
                + (var_1*var_2**2.0*var_3 * (R*S - T) * (f_1 + f_3)) - (var_1**2.0*var_2*var_3 * (S - R*T) * (f_2 + f_3))
                mean = mean_n/v_plus

                # variance function
                var_n = (var_1**2.0*var_2**2.0*var_3**2.0 * (1.0 - R**2 - T**2 - S**2 + 2*R*S*T))
                var = var_n/v_plus
            
        else:

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
