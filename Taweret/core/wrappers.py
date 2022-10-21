"""
Name: wrappers.py
Author: John Yannotty (yannotty.1@osu.edu)
Desc: Defines classes to wrap the BAND mixing and calibration methods. Inspired by bilby documentation 
    (see repo: bilby/bilby/core/sampler/__init__.py) 

Start Date: 10/05/22
Version: 1.0
References:
    https://github.com/lscsoft/bilby/blob/master/bilby/core/sampler/__init__.py

"""

from Taweret.mix.linear import linear_mix as LM
from Taweret.mix.trees import trees_mix as TM
from Taweret.mix.gaussian import bivariate as BM

# Dictionary containing all implemented methods
IMPLEMENTED_MIXERS = {
    "sigmoid": LM,
    "cdf": LM,
    "step": LM,
    "mixbart":TM,
    "bivariate": BM
}

def mixing(model_list, x_exp, y_exp = None, y_err = None, method = 'sigmoid', **kwargs):
    """
        Name: mixing
        Desc: interface for all BAND BMM modules 

        Parameters --- update names 
        ----------
        model_list (list): list of model class instances
        x_exp (np.array): x inputs to train with
        y_exp (np.array): y experimental data to train with
        y_exp (np.array): associated errors with y (std deviations???)
        method (string): name of the mixing method
        **kwargs        : arguments which are mixing method specific 

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
    mixer = mixer_class(model_list, data, method, **kwargs)
    
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


