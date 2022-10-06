"""
Name: wrappers.py
Author: John Yannotty (yannotty.1@osu.edu)
Desc: Defines classes to wrap the BAND mixing and calibration methods
Start Date: 10/05/22
Version: 1.0
"""

from Taweret.mix.linear import linear_mix as LM
from Taweret.mix.trees import trees_mix as TM

IMPLEMENTED_MIXERS = {
    "sigmoid": LM,
    "cdf": LM,
    "step": LM,
    "mixbart":TM
}

def mixing(model_list, x_exp, y_exp = None, y_err = None, method = 'sigmoid', args = {}):
    """
        Name: mixing
        Desc: interface for all BAND BMM modules 

        Parameters --- update names 
        ----------
        model_list (list): list of model class instances
        data (dict): dictionary containing relevant data for the mixing method
            keys: x_exp, y_exp, y_err, (....add as we go)
            values: enter each as a list or np.array, default = None
        method (string): name of the mixing method
        args (dict): arguments which are mixing method specific 

    """
    # Check valid method
    valid_methods = list(IMPLEMENTED_MIXERS.keys())
    if not method.lower() in valid_methods:
        raise ValueError("Invalid mixing method. Valid methods include..." + ", ".join(valid_methods))

    # Check for valid models, must have a predict method
    i = 1
    for m in model_list:
        try:
            getattr(m, 'predict')
        except AttributeError:
            print("Model " + str(i) + " does not have a predict method")
        i+=1

    # Create the data dictionary
    data = {'x_exp':x_exp, 'y_exp':y_exp, 'y_err':y_err}

    # Initialize the mixer class instance
    mixer_class = IMPLEMENTED_MIXERS[method.lower()]
    mixer = mixer_class(model_list, data, method, args)
    
    return mixer



# ---------------------------------------------------------
# ---------------------------------------------------------
# Older checks


# Check valid data (not model specific at this stage)
# data_keys = ['x_exp', 'y_exp', 'y_err'] # add to this list as needed 
# for key in data_keys:
#     if not (key in data.keys()):
#         data.update({key:None})

# # Check if any extra data keys have been passed (not model specific at this stage)
# for in_key in data.keys():
#     if not (in_key in data_keys):
#         raise KeyError("An extra data key was passed. Valid keys include..." + ", ".join(data_keys))




# ---------------------------------------------------------
# ---------------------------------------------------------
# Probably move to base.py

# Mixing Class
class Mixing:
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
        # Check valid method
        valid_methods = ["linear1", "linear2", "linear3"] # change/possibly make dynamic
        if not method in valid_methods:
            raise ValueError("Invalid mixing method. Valid methods include..." + ", ".join(valid_methods))

        # Set the class objects
        self.model_list = model_list
        self.method = method
        self.mixingargs = args
        self.nummodels = len(model_list)

        # --- Add a print statement with requirements and references for the method passed in


        # --- Maybe move data into the fit stage        
        # Set the defaults for the data objects
        data_keys = ['x_exp', 'y_exp', 'y_err']
        for key in data_keys:
            if not (key in data.keys()):
                data.update({key:None})

        # Set the data objects                
        self.x_exp = data['x_exp']
        self.y_exp = data['y_exp']
        self.y_err = data['y_err']

    def fit(self):
        if self.method == "sigmoid":
            pass
        
        if self.method == "cdf":
            pass

        if self.method == "mixbart":
            pass

        

    def predict(self):
        pass

    def weights(self):
        pass

    def plot_weights(self):
        pass

    def plot_mixed_model(self):
        pass

    def mix_likelihood(self):
        pass