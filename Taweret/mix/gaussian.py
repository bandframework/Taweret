#SAMBA methods included here: Bivariate BMM and Trivariate BMM with a GP 
#Written by: Alexandra Semposki
#Authors of SAMBA: Alexandra Semposki, Dick Furnstahl, and Daniel Phillips

#NOTES: Only two model capability as of now

#necessary imports
import numpy as np

class bivariate:

    def __init__(self, x, model_1, model_2, n_model_1=0, n_model_2=0):   #make models into list format

        '''
        Parameters:
        -----------
        x : numpy.linspace
            Input space variable in which mixing is occurring. 

        model_1 : First model with a predict method.

        model_2 : Second model with a predict method. 

        n_model_1 : Number of free parameters in model 1.

        n_model_2 : Number of free parameters in model 2. 

        Returns:
        --------
        None. 
        '''

        #check for predict method in the models     #change later for a list of models
        try:
            getattr(model_1, 'predict')
        except AttributeError:
            print('model 1 does not have a predict method')
        else:
            self.model_1 = model_1

        print(self.model_1)

        try:
            getattr(model_2, 'predict')
        except AttributeError:
            print('model 2 does not have a predict method')
        else:
            self.model_2 = model_2

        self.x = x 
        self.n_model_1 = n_model_1
        self.n_model_2 = n_model_2

        return None


    def mixing_prediction(self, ci=68):  #f_dagger equivalent 

        '''
        The f_dagger function responsible for mixing the two models together
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

        #predict for the two models     #should be changed later for a list of models
        self.prediction_1 = self.model_1.predict(self.x)
        self.prediction_2 = self.model_2.predict(self.x)

        #calculate the models from the class variables 
        f_1 = self.prediction_1[0].flatten()
        f_2 = self.prediction_2[0].flatten()

        v_1 = np.square(self.prediction_1[1]).flatten()
        v_2 = np.square(self.prediction_2[1]).flatten()

        #stacked arrays for mixing
        f = np.vstack([f_1, f_2])
        v = np.vstack([v_1, v_2])

        #initialise arrays
        num = np.zeros(len(self.x))
        denom = np.zeros(len(self.x))
        var = np.zeros(len(self.x))

        #sum over the models in the same input space
        for i in range(len(self.x)):
            num[i] = f[0,i]/v[0,i] + f[1,i]/v[1,i]
            denom[i] = 1/v[0,i] + 1/v[1,i]

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