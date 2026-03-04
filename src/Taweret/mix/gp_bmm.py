# neutron-rich-bmm methods included here: stationary and nonstationary kernel options
# Written by: Alexandra C. Semposki
# Authors of neutron-rich-bmm: Alexandra C. Semposki, Christian Drischler, 
# Dick Furnstahl, and Daniel Phillips

# necessary imports
from Taweret.core.base_mixer import BaseMixer
import numpy as np
import sys
sys.path.append('../Taweret')

# set up classes for each option here
class GPBMM(BaseMixer):   # TODO: figure out how to deal with hyperpriors

    def __init__(self, x, models, mean_function='zero', kernel='rbf', 
                 priors=None):
        '''
        Parameters:
        -----------
        x : numpy.linspace
            Input space variable in which mixing is occurring.

        models : dict
            Dict of models with BaseModel methods.

        mean_function: str
            Selection of the mean function chosen for the GP.
            Choices include: 'zero' and 'spline'.

        kernel: str
            Choice of the kernel to be used in the GP. Choices
            include stationary and non-stationary kernels: 'rbf',
            'matern32', 'matern52', and 'rq' for stationary; 
            'sigmoid', 'tanh', or 'theta' for non-stationary.

        priors: dict
            Dict of hyperpriors for the selected kernel. Default
            priors included in the package will be run if there
            are no specified hyperpriors. 

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

        # str class variables (will become objects later)
        self.mean_function_choice = mean_function
        self.kernel_choice = kernel
        
        # set priors up using defaults if no user given priors
        if priors is None:
            priors = self._default_priors()

        # otherwise use the user provided priors
        self.priors = priors

        # convert models dict() to list
        self.models = [i for i in self.model_dict.values()]
        
        return None
    
    def evaluate(self):
        '''
        Evaluation of the model at a point. Not needed
        for this mixing method at present.
        '''
        return NotImplemented
    
    def evaluate_weights(self):
        '''
        Evaluation of a point estimate of the weights 
        of the models used. Not able to be done for this
        mixing method.
        '''
        return NotImplemented
    
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
    
    # TODO this is where the primary work will be performed
    def predict(self, ci=68):
        '''
        Here the GP needed to perform the mixing is assigned and
        trained on the model means, variances, and covariances
        provided by the chosen model class.

        Parameters:
        -----------
        ci : int, list
            The desired credibility interval(s) (1-sigma, 2-sigma)

        Returns:
        --------
        gp_results : dict
            A dict of the prediction points in the input space,
            and the means, variances, and covariances at each of 
            these specified locations.
        '''

        # credibility interval(s)
        self.ci = ci

        # credibility interval check
        if self.ci == 68:
            val = [1.0]
        elif self.ci == 95:
            val = [1.96]
        elif self.ci == [68, 95]:
            val = [1.0, 1.96]
        else:
            raise ValueError('Choose 1 and/or 2 sigma band.')
        
        # set up the dict of values to return
        gp_results = {
            'x': self.x,
            'mean': self.mean,
            'std': self.std_dev,
            'cov': self.cov
        }

        return gp_results
    
    def predict_weights(self):
        '''
        Predict the weights of the mixed model. Returns
        mean and intervals from the posterior of the
        weights.
        Not needed for this mixing method.
        '''
        return None

    @property
    def prior(self):
        '''
        Return the prior of the parameters in the mixing.
        Not needed for this method.
        '''
        return None

    def prior_predict(self):
        '''
        Find the predicted prior distribution.
        Not implemented for this mixing method.
        '''
        return NotImplemented

    def sample_prior(self):
        '''
        Returns samples from the prior
        distributions for the various weight parameters.
        Not needed for this mixing method.
        '''
        return NotImplemented

    def set_prior(self):
        '''
        Set the priors on the parameters. This will
        be needed for this method but not yet determined how
        to properly use it.
        '''
        return None

    def train(self):
        '''
        Train the GP chosen in the __init__() function
        to optimize its hyperparameters given chosen priors
        and models. Needs to be implemented.
        '''
        return None
