"""
Name: wrappers.py
Author: John Yannotty (yannotty.1@osu.edu)
Desc: Defines base estimator classes which can be used as a the parent class for the specific mixing module
Start Date: 10/05/22
Version: 1.0
"""

from Taweret.core.wrappers import IMPLEMENTED_MIXERS

class BaseMixer:
    # Constructor
    def __init__(self, model_list, data, method, args):
        """
        Parameters
        ----------
        model_list (list): list of model class instances
        data (dict): dictionary containing relevant data for the mixing method
            keys: x_exp, y_exp, y_err, (....add as we go)
            values: enter each as a list or np.array, default = None
        method (string): name of the mixing method
        args (dict): arguments which are mixing method specific 

        """
        # Set the class objects
        self.model_list = model_list
        self.method = method
        self.mixingargs = args
        self.nummodels = len(model_list)

        # Set the data objects                
        self.x_exp = data['x_exp']
        self.y_exp = data['y_exp']
        self.y_err = data['y_err']

    def train(self, prior_info, mcmc_info):   
        print("The model training method has not been implemented for mixing approach = " + self.method)
        train_results = {'fitted_values':None, 'parameters':None, 'posteriors': None }
        return train_results 

    def predict(self, X):
        print("The prediction method has not been implemented for mixing approach = " + self.method)
        self.pred_results = {'pred_values':None, 'posteriors': None }

    def weights(self, X):
        print("The weights method has not been implemented for mixing approach = " + self.method)

    def mix_likelihood(self, X):
        print("The mix_likelihood method has not been implemented for mixing approach = " + self.method)

    def plot_weights(self, X):
        print("The plot_weights method has not been implemented for mixing approach = " + self.method)