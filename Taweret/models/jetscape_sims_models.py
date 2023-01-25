import numpy as np
from Taweret.core.base_model import BaseModel
#from Taweret.utils.utils import normal_log_likelihood_elementwise as log_likelihood_elementwise_utils
import bilby

from functools import lru_cache

#for jetscape models
#import os
import sys
#import pickle
import dill
#import seaborn as sns
import pandas as pd
import math
import matplotlib.pyplot as plt

#from sklearn.decomposition import PCA
#from numpy.linalg import inv
#from sklearn.preprocessing import StandardScaler
#from sklearn.gaussian_process import GaussianProcessRegressor as gpr
#from sklearn.gaussian_process import kernels as krnl
#import scipy.stats as st
#from scipy import optimize
#from scipy.linalg import lapack
#from multiprocessing import Pool
#from multiprocessing import cpu_count

# Path to Jetscape model source code
sys.path.append("/Users/dananjayaliyanage/git/Taweret/subpackages/js-sims-bayes/src")
# Imports from Jetscape code. Need to load the saved emulators.
from configurations import *
from emulator import *


def map_x_to_cent_bins(x : float, obs_to_remove=None):
    """
    Both the simulation and eperiment provide 
    centrality bins instead of a continous centrality
    value. This function Maps a continous value of centrality 
    to discrete centrality bins by finding which bins the provided
    centrality value (x) corresponds to.  

    Paramters
    ---------
    x : float
        Magnitude is between 0 (most central) and 100 (perephiral)

    """
    x = abs(x)
    centr = {}
    for k in obs_cent_list['Pb-Pb-2760']:
        if k in obs_to_remove:
            continue
        val= obs_cent_list['Pb-Pb-2760'][k]

        for i_v, v in enumerate(val):
            lb, ub = v
            if x >= lb and x < ub:
                centr[k]=i_v
                break
        try:
            centr[k]
        except:
            #print(f'centrality : {x} for {k} might be too large. Assign the largest bin avilable {val[-1]}')
            centr[k]= len(val)-1
    return centr


class jetscape_models_pb_pb_2760(BaseModel):
    
    def __init__(self, fix_MAP=True, model_num=0, obs_to_remove=None) -> None:
        """
        Initialize the jetscape emulator model for Pb Pb 
        collisions with Grad viscous corrections.

        Parameters
        ----------
        fix_MAP : bool
            if True, fix the model parameters to MAP values.
        model_number : integer
            0 : Grad, 1 : CE, 3 : PTB
        """
        self._prior = None
        self.fix_MAP = fix_MAP
        self.model_num = model_num
        model_names = {0:'Grad', 1:'CE', 3:'PTB'}
        self.model_name = model_names[model_num]
        self.obs_to_remove=obs_to_remove
        with open(f'{workdir}/emulator/emulator-Pb-Pb-2760-idf-{model_num}.dill',"rb") as f:
            self.Emulators=dill.load(f)
        self.evaluated_MAP = self.Emulators.predict(np.array(MAP_params['Pb-Pb-2760'][self.model_name]).reshape(1,-1), return_cov=True)
    # @lru_cache(maxsize=None)
    # def MAP_eval(self, fix_MAP=True):
    #     "Internal use only to Cache the MAP evaluvation values of emulators"
    #     model_param = np.array(MAP_params['Pb-Pb-2760'][self.model_name])
    #     mn, cov = self.Emulators.predict(model_param.reshape(1,-1), return_cov=True)
    #     return mn, cov

    def evaluate(self, input_values : np.array, model_param = None) -> np.array:
        """
        Predict the mean and error for given input values

        Parameters
        ----------
        input_values : numpy 1darray
            input parameter values.
            centrality, a number betwen 0 (most central) and 100 (perephiral). 
        model_param : numpy 1darray
            value of the model parameter
        """

        if self.fix_MAP:
            model_param = np.array(MAP_params['Pb-Pb-2760'][self.model_name])
        else:
            if model_param is None:
                raise Exception('No model parameters provided. To fix model paramters at MAP, set fix_MAP flag to True')
        x = input_values.flatten()
        mean = []
        var = []

        if len(model_param.flatten()) !=17 :
            raise TypeError('The model_param has to have 17 parameters')
        if self.fix_MAP:
            mn, cov = self.evaluated_MAP
        else:
            mn, cov = self.Emulators.predict(model_param.reshape(1,-1), return_cov=True)

        for xx in x:
            centr = map_x_to_cent_bins(xx, self.obs_to_remove)
            obs=[]
            #ignore cross variances for now.
            obs_var = []
            for k in centr:
                cen_i = centr[k]
                obs.append(mn[k][0][cen_i])
                #print(mn[k].shape)
                obs_var.append((np.diagonal(cov[(k),(k)])[cen_i])[0])
            mean.append(obs)
            var.append(obs_var)

        return np.array(mean), np.sqrt(var)

    #rewrite the log_likelihood_elementwise to take the experimental data types into account.
    #We are going to discard some observables because we do not have experimental measurments for them.
    
    def log_likelihood_elementwise(self, x_exp, y_exp_all, y_err_all, model_param=None):
        """
        Calculate Normal log likelihood elementwise for each centrality in x_exp.

        Parameters
        ----------

        x_exp : 

        y_exp_all : 

        y_err_all :


        """

        predictions, model_errs = self.evaluate(x_exp)
        diff = []
        x_exp = x_exp.flatten()
        if len(x_exp)!=y_exp_all.shape[0]:
            raise Exception(f'Dimensionality mistmach between x_exp and y_exp')
        for i in range(0,len(x_exp)):
            y_exp = y_exp_all[i]
            y_err = y_err_all[i]
            prediction = predictions[i]
            model_err = model_errs[i]

            sigma = np.sqrt(np.square(y_err) + np.square(model_err))
            diff_ar = -0.5* np.square((prediction.flatten() - y_exp)/ sigma) \
                - 0.5 * np.log(2*np.pi)- np.log(sigma)
            diff.append(np.sum(diff_ar))
        return np.array(diff)
        #return log_likelihood_elementwise_utils(self, x_exp, y_exp, y_err, model_param)
    
    def set_prior(self, bilby_priors=None):
        '''
        Set the prior on model parameters.
        For now we will not use any model parameters and fix it to MAP.
        Later we will come back to this. 
        '''
        return None
    #     if bilby_priors is None:
    #         print('Using default priors for model 1')
    #         priors = bilby.prior.PriorDict()
    #         priors['model1_0']=bilby.core.prior.Uniform(1, 6, "model1_0")
    #     else:
    #         priors = bilby_priors
    #     print(priors)
    #     self._prior=priors
    #     return priors

    @property
    def prior(self):
         if self._prior is None:
             return self.set_prior()
         else:
             return self._prior

    @prior.setter
    def prior(self, bilby_prior_dic):
        return self.set_prior(bilby_prior_dic)

from bayes_exp import entry

class exp_data(BaseModel):
    """
    A wrapper for HIC experimental data

    """

    def evaluate(self, input_values : np.array, exp='Pb-Pb-2760', obs_to_remove=None) -> np.array:
        """
        Find the mean and error for experimental observables
        at a given centrality input values.

        Parameters
        ----------
        input_values : numpy 1darray
            centrality (x) values
        """

        mean = []
        sd = []
        for xx in input_values:
            centr = map_x_to_cent_bins(xx, obs_to_remove)
            obs=[]
            #ignore cross variances for now.
            obs_sd = []
            for k in centr:
                cen_i = centr[k]
                obs.append(entry[exp][k]['mean'][0][0][cen_i])
                obs_sd.append(entry[exp][k]['err'][0][0][cen_i])
            mean.append(obs)
            sd.append(obs_sd)

        return np.array(mean), np.array(sd)

    def log_likelihood_elementwise(self):
        return None

    def set_prior(self):
        '''
        Set the prior on any model parameters.
        Not needed for this model. 
        '''
        return None 