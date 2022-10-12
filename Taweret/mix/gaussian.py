#SAMBA methods included here: Multivariate BMM 
#Written by: Alexandra Semposki
#Authors of SAMBA: Alexandra Semposki, Dick Furnstahl, and Daniel Phillips

#NOTES: Only N model capability as of now (no GPs)

#necessary imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

class multivariate:

    def __init__(self, x, models, n_models=0): 

        '''
        Parameters:
        -----------
        x : numpy.linspace
            Input space variable in which mixing is occurring. 

        models : List of models with predict methods. 

        n_models : Number of free parameters per model. 

        Returns:
        --------
        None. 
        '''

        #check for predict method in the models
        for i in range(len(models)):
            try:
                getattr(models[i], 'predict')
            except AttributeError:
                print('model {i} does not have a predict method')
    
        self.models = models
        self.x = x 
        self.n_models = n_models

        return None


    def mixing_prediction(self, ci=68):

        '''
        The f_dagger function responsible for mixing the models together
        in a Gaussian way. 

        Parameters:
        -----------
        None. 

        Returns:
        --------
        mean, var : The mean and variance of the predicted mixed model. 
        '''

        #credibility interval
        self.ci = ci 

        #predict for the two models 
        self.prediction = []

        for i in range(len(self.models)):
            self.prediction.append(self.models[i].predict(self.x))

        #calculate the models from the class variables
        f = []
        v = []
        
        for i in range(len(self.models)):
            f.append(self.prediction[i][0].flatten())
            v.append(np.square(self.prediction[i][1]).flatten())

        #initialise arrays
        num = np.zeros(len(self.x))
        denom = np.zeros(len(self.x))
        var = np.zeros(len(self.x))

        #sum over the models in the same input space
        for i in range(len(self.x)):
            num[i] = np.sum([f[j][i]/v[j][i] for j in range(len(f))])
            denom[i] = np.sum([1/v[j][i] for j in range(len(f))])

        #combine everything via input space tracking 
        mean = num/denom 
        var = 1/denom 

        #credibility interval check
        if self.ci == 68:
            val = 1.0
        elif self.ci == 95:
            val = 1.96
        else:
            raise ValueError('Credibility interval value not found.')

        #calculate intervals
        intervals = np.zeros([len(self.x), 2])
        intervals[:,0] = (mean - val * np.sqrt(var))
        intervals[:,1] = (mean + val * np.sqrt(var))

        return mean, intervals 