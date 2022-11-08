"""
Name: base.py
Author: John Yannotty (yannotty.1@osu.edu)
Desc: Defines base estimator classes which can be used as a the parent class for the specific mixing module
Start Date: 10/05/22
Version: 1.0
"""


class BaseMixer:
    # Constructor
    def __init__(self, model_list, data, method, **kwargs):
        """
        Parameters
        ----------
        model_list (list): list of model class instances
        data (dict): dictionary containing relevant data for the mixing method
            keys: x_exp, y_exp, y_err, (....add as we go OR just use the keys as individual function arguments)
            values: enter each as a list or np.array, default = None
        method (string): name of the mixing method
        args (dict): arguments which are mixing method specific 

        """
        # Set the basic class objects
        self.model_list = model_list
        self.method = method
        self.nummodels = len(model_list)
        
        # Set the data objects                
        self.x_exp = data['x_exp']
        self.y_exp = data['y_exp']
        self.y_err = data['y_err']

        # Initialize the result objects -- objects to be populated by individual mixing methods
        # --Predictions 
        self.pred_mean = None
        self.pred_lb = None
        self.pred_up = None
        # --Weights
        self.wts_mean = None
        self.wts_lb = None
        self.wts_up = None
        # --Error Standard Deviation (when treated as an unknown random variable)
        self.sigma_mean = None
        self.sigma_lower = None
        self.sigma_upper = None

    # Including **kwargs for now, though all arguments should be set in the constrcutor
    def train(self, **kwargs):   
        print("The model training method has not been implemented for mixing approach = " + self.method)
        self.train_results = {'fitted_values':None, 'parameters':None, 'posteriors': None } 

    def predict(self, X):
        print("The prediction method has not been implemented for mixing approach = " + self.method)
        self.pred_results = {'pred_values':None, 'posteriors': None }

    def weights(self, X):
        print("The weights method has not been implemented for mixing approach = " + self.method)
        self.weights_results = {'wt_values':None, 'posteriors': None }

    def mix_likelihood(self, X):
        print("The mix_likelihood method has not been implemented for mixing approach = " + self.method)

    def plot_weights(self, xdim = 0):
        print("The plot_weights method has not been implemented for mixing approach = " + self.method)

    def plot_prediction(self, xdim = 0):
        print("The plot_prediction method has not been implemented for mixing approach = " + self.method)

    # Maybe include various setters and getters 
    # The setters could be useful for changeing arguments wihtout redefining the class ()
    # though with getters you can access the object (since its public)??
    
    # def set_prior_info(): 
        # pass
    # def set_mcmc_info():
        # pass