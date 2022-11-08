"""
Name: trees.py
Author: John Yannotty (yannotty.1@osu.edu)
Desc: Defines the tree mixing class, which is an interface for MixBART 

Start Date: 10/05/22
Version: 1.0
"""
from logging import raiseExceptions
from symbol import pass_stmt
import numpy as np
import matplotlib.pyplot as plt
import sys
import subprocess 
import tempfile 
import shutil
import os

from scipy.stats import norm 
from pathlib import Path 
from scipy.stats import spearmanr 

from Taweret.core.base_mixer import BaseMixer

class Trees(BaseMixer):
    # Overwrite base constructor
    def __init__(self, model_dict, **kwargs):
        # Store model dictionary
        self.model_dict = model_dict
        
        # MCMC Parameters
        self.nummodels = len(model_dict)
        self.ndpost = 1000
        self.nskip = 100
        self.nadapt = 1000
        self.tc = 2
        self.pbd = 0.7
        self.pb = 0.5
        self.stepwpert = 0.1
        self.probchv = 0.1
        self.minnumbot = 5
        self.printevery = 100
        self.numcut = 100
        self.adaptevery = 100
        
        # Set the prior defaults
        self.overallsd = None; self.overallnu = 10
        self.k = 2
        self.ntree = 1; self.ntreeh = 1
        self.power = 2.0
        self.base = 0.95
        self.inform_prior = False
        self.diffwtsprior = False
        self.tauvec = False
        self.betavec = False

        # Define the roots for the output files
        self.xroot = "x"
        self.yroot = "y"
        self.sroot = "s"
        self.chgvroot = "chgv"
        self.froot = "f"
        self.fsdroot = "fsd"
        self.wproot = "wpr"
        self.xiroot = "xi"

        # Set other defaults
        self.modelname = "mixmodel"
        self.summarystats = "FALSE"
        self.local_openbt_path = os.getcwd()

        # Set the kwargs dictionary
        self.__dict__.update((key, value) for key, value in kwargs.items())

        # Set defaults not relevant to model mixing -- only set so cpp code doesn't break               
        self.truncateds = None
        self.modeltype = 9
        # self.ci = 0.68
        self._is_prior_set = False
        self._is_predict_run = False ### Remove ????

    def evaluate(self):
        raise Exception("Not applicable for trees.")

    def evaluate_weights(self):
        raise Exception("Not applicable for trees.")    

    @property
    def map(self):
        return super().map

    # Done
    @property
    def posterior(self):
        return super().posterior

    @property
    # Done
    def prior(self):
        # Init hyperprameters dict
        hyper_params_dict = {}
    
        # Get tau and beta based on the prior cases
        if self.diffwtsprior:
            hyper_params_dict.update({'beta':self.betavec})
            hyper_params_dict.update({'tau':self.tauvec})
        else:
            hyper_params_dict.update({'tau':np.repeat(self.tau,self.nummodels)})
            if self.inform_prior:
                hyper_params_dict.update({'beta':"Not fixed"})
            else:
                hyper_params_dict.update({'beta':np.repeat(self.beta,self.nummodels)})

        # Get the remaining hyperparameters
        hyper_params_dict.update({'nu':self.overallnu, 'lambda':self.overalllambda, 'base':self.base, \
            'power':self.power,'baseh':self.baseh,'powerh':self.powerh})

        return hyper_params_dict

    # DONE
    def set_prior(self, prior_dict):
        # Extract arguments 
        valid_prior_args = ['ntree', 'ntreeh', 'k','power','base','overallsd','overallnu','inform_prior','tauvec', 'betavec'] 
        self._prior = dict()
        for (key, value) in prior_dict.items():
            if key in valid_prior_args:
                self.__dict__.update({key:value}) # Update class dictionary which stores all objects
                self._prior.update({key:value}) # Updates prior specific dictionary

        # Set h arguments
        [self._update_h_args(arg) for arg in ["power", "base",
                                        "pbd", "pb", "stepwpert",
                                        "probchv", "minnumbot"]]

        # Quality check for wtsprior argument
        if "tauvec" in prior_dict.keys() and "betavec" in prior_dict.keys():
            self.diffwtsprior = True
        else:
            self.diffwtsprior = False

        # Run _define_parameters
        self._is_prior_set = True
        self._define_params()

    
    def prior_predict(self):
        return super().prior_predict()


    # DONE
    def train(self,X, y,**kwargs):
        # Cast data to arrays if not already and reshape if needed
        if isinstance(X, list):
            X = np.array(X)

        if len(X.shape) == 1:
            X = X.reshape(X.shape[0],1)

        if isinstance(y, list):
            y = np.array(y)

        if len(y.shape) == 1:
            y = y.reshape(y.shape[0],1)

        # Get number of observations and the number of columns in design matrix
        if X.shape[0] == y.shape[0]:
            self.n = X.shape[0]
            self.p = X.shape[1]
        else:
            raise ValueError("Number of rows in x_exp does not match length of y_exp.")

        # Get predictions from the model set at X's
        fhat_list = []
        shat_list = []
        for m in list(self.model_dict.values()):
            # Get predictions from selected model
            fhat_col, shat_col = m.predict(X)
            
            # Append predictions to respective lists
            fhat_list.append(fhat_col)
            shat_list.append(shat_col)

        # Construct two matrices using concatenate
        f_matrix = np.concatenate(fhat_list, axis = 1)
        s_matrix = np.concatenate(shat_list, axis = 1)
        self.F_train = f_matrix
        self.S_train = s_matrix

        # Set the rest of the data
        #self.y_orig = y ## CLEAN UP
        self.y_train = y
        self.X_train = np.transpose(X) # Reshape X_train to be pxn --- keeping this to remain in sync with remainder of code
        #self.fmean = np.mean(y)

        # Overwrite any default parameters
        if not self._is_prior_set:
            self._define_params()
        
        # Set default outputs
        self.fmean_out = 0
        self.rgy = [np.min(self.y_train), np.max(self.y_train)]
        self.ntreeh = 1

        # Set the mcmc properties from kwargs
        self._set_mcmc_info(kwargs)        

        # Set data informed prior values if needed
        if self.overalllambda is None:
            self.overallsd = np.std(self.y_train, ddof = 1)
            self.overalllambda = self.overallsd**2

        # Write config file
        self._write_config_file()
        self._write_data()
        print("Running model...")
        cmd = "openbtcli"
        self._run_model(cmd)
        
        # See which attributes are returned/can be returned here
        # Return attributes to be saved as a separate fit object:
        res = {} # Missing the influence attribute from the R code (skip for now)
        self.maxx = np.ceil(np.max(self.X_train, axis=1))
        self.minx = np.floor(np.min(self.X_train, axis=1))
        keys = list(self.__dict__.keys()) # Ones we want to save into the object
        for later_key in ['p_test', 'n_test', 'q_lower', 'q_upper', 'xproot', 
            'mdraws', 'sdraws', 'mmean', 'smean', 'msd', 'ssd', 'm_5', 's_5', 
            'm_lower', 's_lower', 'm_upper', 's_upper']:
            if later_key in keys:
                keys.remove(later_key) # Taking away possible residual keys from fitp, fits, or fitv
        for key in keys:
             res[key] = self.__dict__[key]
        res['minx'] = self.minx; res['maxx'] = self.maxx
        
        # Think about this ---
        #self._posterior = self.fpath
        #print("Posterior draws saved to " + self.fpath)
        
        # Get predictions at training points -- more importanlty, get the posterior of sigma
        train_post, train_mean, train_ci, train_sd = self.predict(X, ci = 0.68)
        if self.ntreeh == 1: 
            sigma_post = self.sdraws[:,0]
        else:
            sigma_post = self.sdraws
        
        self._posterior = sigma_post

        #return self._posterior

    # DONE
    def predict(self, X, ci = 0.68):
        # Set q_lower and q_upper
        alpha = (1-ci)
        q_lower = alpha/2
        q_upper = 1-alpha/2
    
        # Casting lists to arrays when needed
        if (type(X) == list):
            X = np.array(X)
        if (len(X.shape) == 1): # If shape is (n, ), change it to (n, 1):
            X = X.reshape(len(X), 1) 
        
        # Get predictions from the model set at X's
        fhat_list = []
        shat_list = []
        for m in list(self.model_dict.values()):
            # Get predictions from selected model
            fhat_col, shat_col = m.predict(X)
            
            # Append predictions to respective lists
            fhat_list.append(fhat_col)
            shat_list.append(shat_col)

        # Construct two matrices using concatenate
        F = np.concatenate(fhat_list, axis = 1)
        self.F_test = F
        #S = np.concatenate(shat_list, axis = 1)
        
        # Set control values
        self.p_test = X.shape[1]
        self.n_test = X.shape[0]
        
        self.q_lower = q_lower; self.q_upper = q_upper
        self.xproot = "xp"
        self.fproot = "fp"
        self._write_chunks(X, (self.n_test) // (self.n_test/(self.tc)),
                            self.xproot,
                            '%.7f')
        
        # Write chunks for f in model mixing
        if self.modeltype == 9:
            self._write_chunks(F, (self.n_test) // (self.n_test/(self.tc)),
                            self.fproot,
                            '%.7f')

        # Set and write config file
        self.configfile = Path(self.fpath / "config.pred")
        pred_params = [self.modelname, self.modeltype,
                    self.xiroot, self.xproot, self.fproot,
                    self.ndpost, self.ntree, self.ntreeh,
                    self.p_test, self.nummodels ,self.tc, self.fmean_out]
        # print(self.ntree); print(self.ntreeh)
        with self.configfile.open("w") as pfile:
            for param in pred_params:
                pfile.write(str(param)+"\n")
        cmd = "openbtpred"
        self._run_model(cmd)
        self._read_in_preds()
        self._is_predict_run = True # mark that the predict function has been called

        # Need to update the results -- one unifed dictionary across all BAND BMM
        pred_post = self.mdraws
        pred_mean = self.pred_mean
        pred_sd = self.pred_sd
        pred_credible_interval = (self.pred_lower, self.pred_upper)
        
        # Return sigma posterior and summary stats
        #sigma_post = self.sdraws
        #sigma_mean = self.sigma_mean
        #sigma_sd = self.sigma_sd
        #sigma_credible_interval = (self.sigma_lower, self.sigma_upper)
        return pred_post, pred_mean, pred_credible_interval, pred_sd
    

    # DONE
    def predict_weights(self, X, ci = 0.68):
        # Set q_lower and q_upper
        alpha = (1-ci)
        q_lower = alpha/2
        q_upper = 1-alpha/2

        # Checks for proper inputs and convert lists to arrays
        if not self.modeltype == 9:
            raise TypeError("Cannot call openbt.mixingwts() method for openbt objects that are not modeltype = 'mixbart'")
        if isinstance(X,list):
            X = np.array(X)
        if (len(X.shape) == 1):
            X = X.reshape(len(X), 1)
        if not self.p == X.shape[1]:
            raise ValueError("The X array does not have the appropriate number of columns.")
        
        # Set control parameters
        self.xwroot = "xw"
        self.fitroot= ".fit"  # default, needed when considering the general class of model mixing problems -- revist this later
        self.q_lower = q_lower
        self.q_upper = q_upper

        # write out the config file
        # Set control values
        self.n_test = X.shape[0] # no need to set this as a class object like in predict function
        self.X_test = X
        self._write_chunks(X, (self.n_test) // (self.n_test/(self.tc)),
                            self.xwroot,
                            '%.7f')

        # Set and write config file
        self.configfile = Path(self.fpath / "config.mxwts")
        pred_params = [self.modelname, self.modeltype,
                       self.xiroot, self.xwroot, self.fitroot,
                       self.ndpost, self.ntree, self.ntreeh,
                       self.p, self.nummodels ,self.tc]
        # print(self.ntree); print(self.ntreeh)
        with self.configfile.open("w") as pfile:
            for param in pred_params:
                pfile.write(str(param)+"\n")
        # run the program
        cmd = "openbtmixingwts"
        self._run_model(cmd)

        # Read in the weights
        self._read_in_wts()
        
        # New: make things a bit more like R, and save attributes to a fit object:
        posterior = self.wdraws 
        post_mean = self.wts_mean 
        post_sd = self.wts_sd
        post_credible_interval = (self.wts_lower,self.wts_upper) 
    
        return posterior, post_mean, post_sd, post_credible_interval


    # -----------------------------------------------------------
    # Plotting
    def plot_prediction(self, xdim = 0):
        col_list = ['red','blue','green','purple','orange']
        if self.pred_mean is None:
            # Compute weights at training points
            print("Getting predictions at training points by default.")
            out_pred = self.predict(self.X_train.transpose())
            self.X_test = self.X_train.transpose()
        
        # Now plot the prediction -- need to improve this plot
        fig = plt.figure(figsize=(6,5))  
        plt.plot(self.X_test[:,xdim], self.pred_mean, color = 'black')
        plt.plot(self.X_test[:,xdim], self.pred_lower, color = 'black', linestyle = "dashed")
        plt.plot(self.X_test[:,xdim], self.pred_upper, color = 'black', linestyle = "dashed")
        for i in range(self.nummodels):
            plt.plot(self.X_test[:,xdim], self.F_test[:,i], color = col_list[i], linestyle = 'dotted')
        plt.scatter(self.X_train[xdim,:] ,self.y_train) # Recall X_train was transposed in the beginning 
        plt.title("Posterior Mean Prediction")
        plt.xlabel("X") # Update Label
        plt.ylabel("F(X)") # Update Label 
        plt.grid(True, color='lightgrey')
        plt.show()


    def plot_weights(self, xdim = 0):
        # Check if weights are already loaded
        col_list = ['red','blue','green','purple','orange']
        if self.wts_mean is None:
            # Compute weights at training points
            print("Computing weights at training points by default.")
            out_wts = self.weights(self.X_train.transpose())
            self.X_test = self.X_train.transpose()
        
        # Now plot the weights -- need to improve this plot
        fig = plt.figure(figsize=(6,5))  
        for i in range(self.nummodels):
            plt.plot(self.X_test[:,xdim], self.wts_mean[:,i], color = col_list[i])
            plt.plot(self.X_test[:,xdim], self.wts_lower[:,i], color = col_list[i], linestyle = "dashed")
            plt.plot(self.X_test[:,xdim], self.wts_upper[:,i], color = col_list[i], linestyle = "dashed")
        plt.title("Posterior Weight Functions")
        plt.xlabel("X") # Update Label
        plt.ylabel("W(X)") # Update Label 
        plt.grid(True, color='lightgrey')
        plt.show()


    def plot_sigma(self):
        fig = plt.figure(figsize=(6,5))
        plt.hist(self.posterior, zorder = 2)
        plt.title("Posterior Weight Functions")
        plt.xlabel("X") # Update Label
        plt.ylabel("W(X)") # Update Label 
        plt.grid(True, color='lightgrey', zorder = 0)
        plt.show()

    # ----------------------------------------------------------
    # "Private" Functions
    def _define_params(self):
        """Set up parameters for the openbtcli
        """
        # Overwrite the hyperparameter settings when model mixing
        if self.inform_prior:
            self.tau = 1/(2*self.ntree*self.k)
            self.beta = 1/self.ntree
        else:
            self.tau = 1/(2*np.sqrt(self.ntree)*self.k)
            self.beta = 1/(2*self.ntree)
        
        # overall lambda calibration
        if not self.overallsd is None:
            self.overalllambda = self.overallsd**2
        else:
            self.overalllambda = None
        
        # Set overall nu 
        if self.overallnu is None:
            self.overallnu = 10


    def _read_in_preds(self):
        mdraw_files = sorted(list(self.fpath.glob(self.modelname+".mdraws*")))
        sdraw_files = sorted(list(self.fpath.glob(self.modelname+".sdraws*")))
        mdraws = []
        for f in mdraw_files:
            read = open(f, "r"); lines = read.readlines()
            if lines[0] != '\n' and lines[1] != '\n': # If it's nonempty
                 mdraws.append(np.loadtxt(f))
        # print(mdraws[0].shape); print(len(mdraws))
        self.mdraws = np.concatenate(mdraws, axis=1) # Got rid of the transpose
        sdraws = []
        for f in sdraw_files:
            read = open(f, "r"); lines = read.readlines()
            if lines[0] != '\n' and lines[1] != '\n': # If it's nonempty
                 sdraws.append(np.loadtxt(f))
        # print(sdraws[0]); print(sdraws[0][0])
        # print(len(sdraws)); print(len(sdraws[0])); print(len(sdraws[0][0]))
        self.sdraws = np.concatenate(sdraws, axis=1) # Got rid of the transpose
        
        # New (added by me), since R returns arrays like these by default:
        # Calculate mmean and smean arrays, and related statistics
        self.pred_mean = np.empty(len(self.mdraws[0]))
        self.sigma_mean = np.empty(len(self.sdraws[0]))
        self.pred_sd = np.empty(len(self.mdraws[0]))
        self.sigma_sd = np.empty(len(self.mdraws[0]))
        self.pred_5 = np.empty(len(self.mdraws[0]))
        self.sigma_5 = np.empty(len(self.mdraws[0]))
        self.pred_lower = np.empty(len(self.mdraws[0]))
        self.sigma_lower = np.empty(len(self.sdraws[0]))
        self.pred_upper = np.empty(len(self.mdraws[0]))
        self.sigma_upper = np.empty(len(self.sdraws[0]))
        for j in range(len(self.mdraws[0])):
             self.pred_mean[j] = np.mean(self.mdraws[:, j])
             self.sigma_mean[j] = np.mean(self.sdraws[:, j])
             self.pred_sd[j] = np.std(self.mdraws[:, j], ddof = 1)
             self.sigma_sd[j] = np.std(self.sdraws[:, j], ddof = 1)
             self.pred_5[j] = np.quantile(self.mdraws[:, j], 0.50)
             self.sigma_5[j] = np.quantile(self.sdraws[:, j], 0.50)
             self.pred_lower[j] = np.quantile(self.mdraws[:, j], self.q_lower)
             self.sigma_lower[j] = np.quantile(self.sdraws[:, j], self.q_lower)
             self.pred_upper[j] = np.quantile(self.mdraws[:, j], self.q_upper)
             self.sigma_upper[j] = np.quantile(self.sdraws[:, j], self.q_upper)


    def _read_in_wts(self):
        # Initialize the wdraws dictionary
        self.wdraws = {}        

        # Initialize summary statistic matrices for the wts
        self.wts_mean = np.empty((self.n_test,self.nummodels))
        self.wts_sd = np.empty((self.n_test,self.nummodels))
        self.wts_5 = np.empty((self.n_test,self.nummodels))
        self.wts_lower = np.empty((self.n_test,self.nummodels))
        self.wts_upper = np.empty((self.n_test,self.nummodels))

        # Get the weight files
        for k in range(self.nummodels):
            wdraw_files = sorted(list(self.fpath.glob(self.modelname+".w"+str(k+1)+"draws*")))
            wdraws = []
            for f in wdraw_files:
                read = open(f, "r"); lines = read.readlines()
                if lines[0] != '\n' and lines[1] != '\n': # If it's nonempty
                    wdraws.append(np.loadtxt(f))

            # Store the wdraws array in the self.wdraws dictionary under the key wtname  
            wtname = "w" + str(k+1)
            self.wdraws[wtname] = np.concatenate(wdraws, axis=1) # Got rid of the transpose
            
            for j in range(len(self.wdraws[wtname][0])):
                self.wts_mean[j][k] = np.mean(self.wdraws[wtname][:, j])
                self.wts_sd[j][k] = np.std(self.wdraws[wtname][:, j], ddof = 1)
                self.wts_5[j][k] = np.quantile(self.wdraws[wtname][:, j], 0.50)
                self.wts_lower[j][k] = np.quantile(self.wdraws[wtname][:, j], self.q_lower)
                self.wts_upper[j][k] = np.quantile(self.wdraws[wtname][:, j], self.q_upper)

    
    def _run_model(self, cmd="openbtcli"):
        # Check to see if executable is in the current directory
        sh = shutil.which(cmd)
    
        # Execute the subprocess, changing directory when needed
        if sh is None:        
            # openbt exe were not found in the current directory -- try the local directory passed in
            sh = shutil.which(cmd, path = self.local_openbt_path) 
            if sh is None:
                raise FileNotFoundError("Cannot find openbt executables. Please specify the path using the argument local_openbt_path in the constructor.")
            else:
                cmd = sh
                sp = subprocess.run(["mpirun", "-np", str(self.tc), cmd, str(self.fpath)],
                                stdin=subprocess.DEVNULL, capture_output=True)    
        else:
            sp = subprocess.run(["mpirun", "-np", str(self.tc), cmd, str(self.fpath)],
                                stdin=subprocess.DEVNULL, capture_output=True)


    def _set_mcmc_info(self, mcmc_dict):
        # Extract arguments
        valid_mcmc_args = ["ndpost","nskip","nadapt","tc","pbd","pb","stepwpert","probchv","minnumbot","printevery","numcut", "adaptevery","xicuts"]
        for (key, value) in mcmc_dict.items():
            if key in valid_mcmc_args:
                self.__dict__.update({key:value})

        # Cutpoints
        if "xicuts" not in self.__dict__:
            self.xi = {}
            maxx = np.ceil(np.max(self.X_train, axis=1))
            minx = np.floor(np.min(self.X_train, axis=1))
            for feat in range(self.p):
                xinc = (maxx[feat] - minx[feat])/(self.numcut+1)
                self.xi[feat] = [
                    np.arange(1, (self.numcut)+1)*xinc + minx[feat]]

        # Birth and Death probability -- set product tree pbd to 0 for selected models 
        if (isinstance(self.pbd, float)):
            self.pbd = [self.pbd, 0] 
        
        # Update the product tree arguments
        [self._update_h_args(arg) for arg in ["pbd", "pb", "stepwpert",
                                              "probchv", "minnumbot"]]


    def _set_wts_prior(self, betavec, tauvec):
        # Cast lists to np.arrays when needed
        if isinstance(betavec, list):
            betavec = np.array(betavec)
        if isinstance(tauvec, list):
            tauvec = np.array(tauvec)

        # Check lengths
        if not (len(tauvec) == self.nummodels and len(betavec) == self.nummodels):
            raise ValueError("Incorrect vector length for tauvec and/or betavec. Lengths must be equal to the number of models.")

        # Store the hyperparameters passed in 
        self.diffwtsprior = True
        self.betavec = betavec
        self.tauvec = tauvec


    def _update_h_args(self, arg):
        try:
            self.__dict__[arg + "h"] = self.__dict__[arg][1]
            self.__dict__[arg] = self.__dict__[arg][0]
        except:
            self.__dict__[arg + "h"] = self.__dict__[arg]
    
    

    # Need to generalize -- this is only used in fit 
    def _write_config_file(self):
        """Create temp directory to write config and data files
        """
        f = tempfile.mkdtemp(prefix="openbtpy_")
        self.fpath = Path(f)
        run_params = [self.modeltype,
                      self.xroot, self.yroot, self.fmean_out,
                      self.ntree, self.ntreeh,
                      self.ndpost, self.nskip,
                      self.nadapt, self.adaptevery,
                      self.tau, self.beta,
                      self.overalllambda,
                      self.overallnu, self.base,
                      self.power, self.baseh, self.powerh,
                      self.tc, self.sroot, self.chgvroot,
                      self.froot, self.fsdroot, self.inform_prior,
                      self.wproot, self.diffwtsprior,
                      self.pbd, self.pb, self.pbdh, self.pbh, self.stepwpert,
                      self.stepwperth,
                      self.probchv, self.probchvh, self.minnumbot,
                      self.minnumboth, self.printevery, self.xiroot, self.modelname,
                      self.summarystats]
        # print(run_params)
        self.configfile = Path(self.fpath / "config")
        with self.configfile.open("w") as tfile:
            for param in run_params:
                tfile.write(str(param)+"\n")


    def _write_chunks(self, data, no_chunks, var, *args):
        if no_chunks == 0:
             print("Writing all data to one 'chunk'"); no_chunks = 1
        if (self.tc - int(self.tc) == 0):
             splitted_data = np.array_split(data, no_chunks)
        else:
             sys.exit('Fit: Invalid tc input - exiting process')   
        int_added = 0 if var in ["xp","fp","xw"] else 1

        for i, ch in enumerate(splitted_data):
             np.savetxt(str(self.fpath / Path(self.__dict__[var+"root"] + str(i+int_added))),
               ch, fmt=args[0])


    def _write_data(self):
        splits = (self.n - 1) // (self.n/(self.tc)) # Should = tc - 1 as long as n >= tc
        # print("splits =", splits)
        self._write_chunks(self.y_train, splits, "y", '%.7f')
        self._write_chunks(np.transpose(self.X_train), splits, "x", '%.7f')
        self._write_chunks(np.ones((self.n), dtype="int"),
                            splits, "s", '%.0f')
        print(self.fpath)
        if self.X_train.shape[0] == 1:
             #print("1 x variable, so correlation = 1")
             np.savetxt(str(self.fpath / Path(self.chgvroot)), [1], fmt='%.7f')
        elif self.X_train.shape[0] == 2:
             #print("2 x variables")
             np.savetxt(str(self.fpath / Path(self.chgvroot)),
                        [spearmanr(self.X_train, axis=1)[0]], fmt='%.7f')
        else:
             #print("3+ x variables")
             np.savetxt(str(self.fpath / Path(self.chgvroot)),
                        spearmanr(self.X_train, axis=1)[0], fmt='%.7f')
             
        for k, v in self.xi.items():
            np.savetxt(
                str(self.fpath / Path(self.xiroot + str(k+1))), v, fmt='%.7f')
        
        # Write model mixing files
        if self.modeltype == 9:
            # F-hat matrix
            self._write_chunks(self.F_train, splits, "f", '%.7f')
            # S-hat matrix when using inform_prior
            if self.inform_prior:
                self._write_chunks(self.S_train, splits, "fsd", '%.7f')
            # Wts prior when passed in
            if self.diffwtsprior:
                np.savetxt(str(self.fpath / Path(self.wproot)), np.concatenate(self.betavec, self.tauvec),fmt='%.7f')
        

    # --------------------------------------------
    # Workking -- trees jail
    # Return predictions of sigma --- need to figure this out (may group this into _posterior??)
    # NEED TO THINK ABOUT THIS
    def predict_sigma(self, X, ci = 0.68):
        if self._is_predict_run and self.X_test == X:
            # If prediction was run and test data is the same
            sigma_post = self.sdraws
            sigma_mean = self.sigma_mean
            sigma_sd = self.sigma_sd
            sigma_credible_interval = (self.sigma_lower, self.sigma_upper)
        
            # Get new credible intervals if ci level changes
            if not self.pred_ci == ci:
                new_sigma_lower = np.empty(len(self.sdraws[0]))
                new_sigma_upper = np.empty(len(self.sdraws[0])) 
                for j in range(len(self.sdraws[0])):
                    new_sigma_lower[j] = np.quantile(self.sdraws[:, j], self.q_lower)
                    new_sigma_upper[j] = np.quantile(self.sdraws[:, j], self.q_upper)
                sigma_credible_interval = (new_sigma_lower, new_sigma_upper)
        else:
            # Run predict at X if predict has not already been called
            _,_,_,_ = self.predict(X, ci)
            sigma_post = self.sdraws
            sigma_mean = self.sigma_mean
            sigma_sd = self.sigma_sd
            sigma_credible_interval = (self.sigma_lower, self.sigma_upper)        
        
        return sigma_post, sigma_mean, sigma_credible_interval, sigma_sd
    