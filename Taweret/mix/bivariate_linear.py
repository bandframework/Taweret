import numpy as np
import math
from ..utils.utils import log_likelihood_elementwise, mixture_function, eps
import matplotlib.pyplot as plt
from ..core import base_mixer, base_model

class BivariateLinear(base_mixer):

    '''
    Local linear mixing of two models.

    '''

    def __init__(self, models_dic, method='sigmoid',
                nargs_for_each_model=[], n_mix=0):
        '''
        Parameters
        ----------
        models_dic : dictionary {'name' : model1, 'name' : model2}
            Two models to mix, each must be derived from the base_model.
        method : str
            mixing function
        nargs_for_each_model : list
            number of free parameters for each model
        n_mix : int
            number of free parameters in the mixing funtion
        
        '''
        # check if more than two models are trying to be mixed
        if len(models_dic)!==2:
            raise Exception('Bivariate linear mixing requires only two models.\
                            Please look at the other mixing methods in Taweret \
                            for multi modal mixing')
        # check that lengths of lists are compatible
        if len(models_dic) != len(nargs_for_each_model) and len(nargs_for_each_model) != 0:
            raise Exception("""in linear_mix.__init__: len(nargs_for_each_model) = 0 for
                            learning weights only 
                            otherwise len(nargs_for_each_model) = len(models_dic)""")

        #check for predict method in the models
        for i, model in enumerate(models_dic.items()):
            try:
                issubclass(model, base_model)
            except AttributeError:
                print(f'model {models_dic.keys()[i]} is not derived from core.base_model class')
            else:
                continue

        self.models_dic = models_dic

        #check if the mixing method exist
        if method not in ['step', 'sigmoid', 'cdf', 'switchcos']:
            raise Exception('Specefied mixing function is not found')

        self.method = method
        self.nargs_for_each_model = nargs_for_each_model
        self.n_mix = n_mix

        # function returns
