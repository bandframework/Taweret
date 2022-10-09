"""
Name: trees.py
Author: John Yannotty (yannotty.1@osu.edu)
Desc: Defines the tree mixing class, which is an interface for MixBART 

Start Date: 10/05/22
Version: 1.0
"""
import numpy as np
import sys
import subprocess 
import tempfile 
import shutil
import os
from scipy.stats import norm 
from pathlib import Path 
from scipy.stats import spearmanr 

from Taweret.core.base import BaseMixer

class trees_mix(BaseMixer):
    # Overwrite base constructor
    def __init__(self, model_list, data, method, **kwargs):
        # Call the base constructor
        BaseMixer.__init__(self, model_list, data, method, **kwargs)
        
        # Set the defaults for the data objects
        tree_data_keys = ['x_exp', 'y_exp'] 
        for k in tree_data_keys:
            if data[k] is None:
                raise ValueError("Argument " + k + " is None.")

        # Cast data to arrays if not already and reshape if needed
        if isinstance(data['x_exp'], list):
            data['x_exp'] = np.array(data['x_exp'])

        if len(data['x_exp'].shape) == 1:
            data['x_exp'] = data['x_exp'].reshape(data['x_exp'].shape[0],1)

        if isinstance(data['y_exp'], list):
            data['y_exp'] = np.array(data['y_exp'])

        if len(data['y_exp'].shape) == 1:
            data['y_exp'] = data['y_exp'].reshape(data['y_exp'].shape[0],1)

        # Get number of observations and the number of columns in design matrix
        if data['x_exp'].shape[0] == data['y_exp'].shape[0]:
            self.n = data['x_exp'].shape[0]
            self.p = data['x_exp'].shape[1]
        else:
            raise ValueError("Number of rows in x_exp does not match length of y_exp.")

        # Get predictions from the model set at X's
        fhat_list = []
        shat_list = []
        for m in model_list:
            # Get predictions from selected model
            fhat_col, shat_col = m.predict(data['x_exp'])
            
            # Append predictions to respective lists
            fhat_list.append(fhat_col)
            shat_list.append(shat_col)

        # Construct two matrices using concatenate
        f_matrix = np.concatenate(fhat_list, axis = 1)
        s_matrix = np.concatenate(shat_list, axis = 1)
        self.F_train = f_matrix
        self.S_train = s_matrix

        # Set the rest of the data
        self.y_orig = data['y_exp']
        self.X_train = np.transpose(data['x_exp']) # Reshape X_train to be pxn --- keeping this to remain in sync with remainder of code
        self.fmean = np.mean(data['y_exp'])

        # Set default values for prior and mcmc related parameters -- maybe move into constructor
        # MCMC Parameters
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
        self.nsprior = True
        self.wtsprior = False
        self.tauvec = False
        self.betavec = False

        # Set other defaults
        self.modelname = "mixmodel"
        self.summarystats = "FALSE"
        self.local_openbt_path = os.getcwd()

        # Set the kwargs dictionary
        self.__dict__.update((key, value) for key, value in kwargs.items())

        # Set defaults not relevant to model mixing -- only set so cpp code doesn't break               
        self.truncateds = None
        self.modeltype = 9
        

    def train(self, **kwargs):
        # Overwrite any default parameters
        self._define_params()
        print("Writing config file and data")
        self._write_config_file()
        self._write_data()
        print("Running model...")
        cmd = "openbtcli"
        self._run_model(cmd)
        
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
        return res

    def predict(self, X, q_lower, q_upper):
        # Casting lists to arrays when needed
        if (type(X) == list):
            X = np.array(X)
            print("Completed list-to-numpy_array conversion for the preds. Be careful about row/column mixups!")
        if (len(X.shape) == 1): # If shape is (n, ), change it to (n, 1):
            X = X.reshape(len(X), 1) 
        
        # Get predictions from the model set at X's
        fhat_list = []
        shat_list = []
        for m in self.model_list:
            # Get predictions from selected model
            fhat_col, shat_col = m.predict(X)
            
            # Append predictions to respective lists
            fhat_list.append(fhat_col)
            shat_list.append(shat_col)

        # Construct two matrices using concatenate
        F = np.concatenate(fhat_list, axis = 1)
        #S = np.concatenate(shat_list, axis = 1)
        
        # Set control values
        self.p_test = X.shape[1]
        self.n_test = X.shape[0]
        self.q_lower = q_lower; self.q_upper = q_upper
        self.xproot = "xp"
        self.fproot = "fp"
        self.__write_chunks(X, (self.n_test) // (self.n_test/(self.tc)),
                            self.xproot,
                            '%.7f')
        
        # Write chunks for f in model mixing
        if self.modeltype == 9:
            self.__write_chunks(F, (self.n_test) // (self.n_test/(self.tc)),
                            self.fproot,
                            '%.7f')

        # Set and write config file
        self.configfile = Path(self.fpath / "config.pred")
        pred_params = [self.modelname, self.modeltype,
                       self.xiroot, self.xproot, self.fproot,
                       self.ndpost, self.ntree, self.ntreeh,
                       self.p_test, self.nummodels ,self.tc, self.fmean]
        # print(self.ntree); print(self.ntreeh)
        with self.configfile.open("w") as pfile:
            for param in pred_params:
                pfile.write(str(param)+"\n")
        cmd = "openbtpred"
        self._run_model(cmd)
        self._read_in_preds()
        
        # Need to update the results -- one unifed dictionary across all BAND BMM
        res = {}
        res['mdraws'] = self.mdraws; res['sdraws'] = self.sdraws;
        res['mmean'] = self.mmean; res['smean'] = self.smean;
        res['msd'] = self.msd; res['ssd'] = self.ssd;
        res['m_5'] = self.m_5; res['s_5'] = self.s_5;
        res['m_lower'] = self.m_lower; res['s_lower'] = self.s_lower;
        res['m_upper'] = self.m_upper; res['s_upper'] = self.s_upper;
        res['q_lower'] = self.q_lower; res['q_upper'] = self.q_upper;
        res['x_test'] = X; res['modeltype'] = self.modeltype
        return res

    def weights(self, X, q_lower, q_upper):
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
        self.__write_chunks(X, (self.n_test) // (self.n_test/(self.tc)),
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
        res = {}
        res['wdraws'] = self.wdraws 
        res['wmean'] = self.wmean 
        res['wsd'] = self.wsd
        res['w_5'] = self.w_5
        res['w_lower'] = self.w_lower
        res['w_upper'] = self.w_upper
        res['q_lower'] = self.q_lower; res['q_upper'] = self.q_upper
        res['x_test'] = X; res['modeltype'] = self.modeltype
        return res


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
        self.mmean = np.empty(len(self.mdraws[0]))
        self.smean = np.empty(len(self.sdraws[0]))
        self.msd = np.empty(len(self.mdraws[0]))
        self.ssd = np.empty(len(self.mdraws[0]))
        self.m_5 = np.empty(len(self.mdraws[0]))
        self.s_5 = np.empty(len(self.mdraws[0]))
        self.m_lower = np.empty(len(self.mdraws[0]))
        self.s_lower = np.empty(len(self.sdraws[0]))
        self.m_upper = np.empty(len(self.mdraws[0]))
        self.s_upper = np.empty(len(self.sdraws[0]))
        for j in range(len(self.mdraws[0])):
             self.mmean[j] = np.mean(self.mdraws[:, j])
             self.smean[j] = np.mean(self.sdraws[:, j])
             self.msd[j] = np.std(self.mdraws[:, j], ddof = 1)
             self.ssd[j] = np.std(self.sdraws[:, j], ddof = 1)
             self.m_5[j] = np.percentile(self.mdraws[:, j], 0.50)
             self.s_5[j] = np.percentile(self.sdraws[:, j], 0.50)
             self.m_lower[j] = np.percentile(self.mdraws[:, j], self.q_lower)
             self.s_lower[j] = np.percentile(self.sdraws[:, j], self.q_lower)
             self.m_upper[j] = np.percentile(self.mdraws[:, j], self.q_upper)
             self.s_upper[j] = np.percentile(self.sdraws[:, j], self.q_upper)


    def _read_in_wts(self):
        # Initialize the wdraws dictionary
        self.wdraws = {}        

        # Initialize summary statistic matrices for the wts
        self.wmean = np.empty((self.n_test,self.nummodels))
        self.wsd = np.empty((self.n_test,self.nummodels))
        self.w_5 = np.empty((self.n_test,self.nummodels))
        self.w_lower = np.empty((self.n_test,self.nummodels))
        self.w_upper = np.empty((self.n_test,self.nummodels))


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
                self.wmean[j][k] = np.mean(self.wdraws[wtname][:, j])
                self.wsd[j][k] = np.std(self.wdraws[wtname][:, j], ddof = 1)
                self.w_5[j][k] = np.percentile(self.wdraws[wtname][:, j], 0.50)
                self.w_lower[j][k] = np.percentile(self.wdraws[wtname][:, j], self.q_lower)
                self.w_upper[j][k] = np.percentile(self.wdraws[wtname][:, j], self.q_upper)


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
        self.wtsprior = True
        self.betavec = betavec
        self.tauvec = tauvec



    def _update_h_args(self, arg):
        try:
            self.__dict__[arg + "h"] = self.__dict__[arg][1]
            self.__dict__[arg] = self.__dict__[arg][0]
        except:
            self.__dict__[arg + "h"] = self.__dict__[arg]
    

    def _define_params(self):
        """Set up parameters for the openbtcli
        """
        # Can simplify this if needed
        if (self.modeltype in [4, 5, 8]):
           self.y_train = self.y_orig - self.fmean
           self.fmean_out = 0
           self.rgy = [np.min(self.y_train), np.max(self.y_train)]
        elif (self.modeltype in [6, 7]):
            self.fmean_out = norm.ppf(self.fmean)
            self.y_train = self.y_orig
            self.rgy = [-2, 2]
            self.uniqy = np.unique(self.y_train) # Already sorted, btw
            if(len(self.uniqy) > 2 or self.uniqy[1] != 0 or self.uniqy[2] != 1):
                 sys.exit("Invalid y.train: Probit requires dichotomous response coded 0/1") 
        elif self.modeltype in [9]:
            self.fmean_out = 0
            self.fmean = 0
            self.y_train = self.y_orig
            self.rgy = [np.min(self.y_train), np.max(self.y_train)]
            self.ntreeh = 1
        else: # Unused modeltypes for now, but still set their properties just in case
            self.y_train = self.y_orig 
            self.fmean_out = None
            self.rgy = [-2, 2] # These proprties are ambiguous for these modeltypes by the way...
            
        #self.n = self.y_train.shape[0]
        #self.p = self.X_train.shape[0]
        
        # Cutpoints
        if "xicuts" not in self.__dict__:
            self.xi = {}
            maxx = np.ceil(np.max(self.X_train, axis=1))
            minx = np.floor(np.min(self.X_train, axis=1))
            for feat in range(self.p):
                xinc = (maxx[feat] - minx[feat])/(self.numcut+1)
                self.xi[feat] = [
                    np.arange(1, (self.numcut)+1)*xinc + minx[feat]]
        
        # Set the terminal node hyperparameters
        self.tau = (self.rgy[1] - self.rgy[0])/(2*np.sqrt(self.ntree)*self.k)

        # Overwrite the hyperparameter settings when model mixing
        if self.modeltype == 9:
            if self.nsprior:
                self.tau = 1/(2*self.ntree*self.k)
                self.beta = 1/self.ntree
            else:
                self.tau = 1/(2*np.sqrt(self.ntree)*self.k)
                self.beta = 1/(2*self.ntree)
        else:
            self.beta = 0

        # Map for the overall sd default values
        osd = np.std(self.y_train, ddof = 1)

        # Overall sd update
        if (self.overallsd is None):
             print("Overwriting overallsd to agree with the model's default")
             self.overallsd = osd
        
        # overall lambda calibration
        self.overalllambda = self.overallsd**2
        
        # Set overall nu 
        if self.overallnu is None:
            self.overallnu = 10

        # Birth and Death probability -- set product tree pbd to 0 for selected models 
        if (isinstance(self.pbd, float)):
            self.pbd = [self.pbd, 0] 
        
        [self._update_h_args(arg) for arg in ["power", "base",
                                              "pbd", "pb", "stepwpert",
                                              "probchv", "minnumbot"]]
        # define the roots for the output files
        self.xroot = "x"
        self.yroot = "y"
        self.sroot = "s"
        self.chgvroot = "chgv"
        self.froot = "f"
        self.fsdroot = "fsd"
        self.wproot = "wpr"
        self.xiroot = "xi"
        


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
                      self.froot, self.fsdroot, self.nsprior,
                      self.wproot, self.wtsprior,
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


    def __write_chunks(self, data, no_chunks, var, *args):
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
        self.__write_chunks(self.y_train, splits, "y", '%.7f')
        self.__write_chunks(np.transpose(self.X_train), splits, "x", '%.7f')
        self.__write_chunks(np.ones((self.n), dtype="int"),
                            splits, "s", '%.0f')
        print(self.fpath)
        if self.X_train.shape[0] == 1:
             print("1 x variable, so correlation = 1")
             np.savetxt(str(self.fpath / Path(self.chgvroot)), [1], fmt='%.7f')
        elif self.X_train.shape[0] == 2:
             print("2 x variables")
             np.savetxt(str(self.fpath / Path(self.chgvroot)),
                        [spearmanr(self.X_train, axis=1)[0]], fmt='%.7f')
        else:
             print("3+ x variables")
             np.savetxt(str(self.fpath / Path(self.chgvroot)),
                        spearmanr(self.X_train, axis=1)[0], fmt='%.7f')
             
        for k, v in self.xi.items():
            np.savetxt(
                str(self.fpath / Path(self.xiroot + str(k+1))), v, fmt='%.7f')
        
        # Write model mixing files
        if self.modeltype == 9:
            # F-hat matrix
            self.__write_chunks(self.F_train, splits, "f", '%.7f')
            # S-hat matrix when using nsprior
            if self.nsprior:
                self.__write_chunks(self.S_train, splits, "fsd", '%.7f')
            # Wts prior when passed in
            if self.wtsprior:
                np.savetxt(str(self.fpath / Path(self.wproot)), np.concatenate(self.betavec, self.tauvec),fmt='%.7f')
        

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
        

    #--------------------------------------------
    # Extra setters right now -- could be useful for user, could be assigned _ tag to denote internal use, could be removed - not sure yet 
    def set_prior_info(self, **kwargs):
        # Extract arguments 
        valid_prior_args = ['ntree', 'ntreeh', 'k','power','base','overallsd','overallnu','nsprior','tauvec', 'betavec'] 
        for (key, value) in kwargs.itmes():
            if key in valid_prior_args:
                self.__dict__.update({key:value})

        # Quality check for wtsprior argument
        if "tauvec" in kwargs.keys() and "betavec" in kwargs.keys():
            self.wtsprior = True
        else:
            self.wtsprior = False

    def set_prior_info(self, **kwargs):
        # Extract arguments
        valid_mcmc_args = ["ndpost","nskip","nadapt","tc","pbd","pb","stepwpert","probchv","minnumbot","printevery","numcut", "adaptevery"]
        for (key, value) in kwargs.itmes():
            if key in valid_mcmc_args:
                self.__dict__.update({key:value})
    


#------------------------------------------------
# Testing MixBART
#------------------------------------------------
# Polynomial function
# import matplotlib.pyplot as plt
# class FP:
#     def __init__(self,a=0,b=0,c=1,p=1):
#         self.a = a
#         self.b = b
#         self.c = c
#         self.p = p
    
#     def predict(self, x):
#         if isinstance(x, list):
#             x = np.array(x)
#         m = self.c*(x-self.a)**self.p + self.b
#         if len(m.shape) == 1:
#             m = m.reshape(m.shape[0],1) 
#         s = np.array([1]*x.shape[0]).reshape(m.shape[0],1)
#         return m,s

# model_list = [FP(0,-2,4,1), FP(0,2,-4,1)]

# # Training Data
# n_train = 15
# n_test = 100
# s = 0.1

# x_train = np.concatenate([np.array([0.01,0.1,0.25]),np.linspace(0.45,1.0, n_train-3)])
# x_test = np.linspace(0.01, 1.0, n_test)

# np.random.seed(1234567)
# fp = FP(0.5,0,8,2)
# f0_train,_ = fp.predict(x_train)
# f0_test,_ = fp.predict(x_test)
# y_train = f0_train + np.random.normal(0,s,n_train).reshape(n_train,1)

# data = {'x_exp':x_train, 'y_exp': y_train, 'y_err':None}
# method = 'mixbart'
# mix = trees_mix(model_list = model_list, data = data, method = method, tc = 4, modelname = "parabola", ntree = 10, k = 1, ndpost = 10000, nskip = 2000, nadapt = 5000, 
#                 adaptevery = 500, overallsd = 0.556, minnumbot = 1, local_openbt_path = "/home/johnyannotty/Documents/Open BT Project SRC")

# fitx = mix.train()
# fitxp = mix.predict(X = x_test, q_lower=0.025, q_upper=0.975)
# fitxw = mix.weights(X = x_test, q_lower=0.025, q_upper=0.975)


# # Get the data for simplicty
# fp1 = FP(0,-2,4,1)
# fp2 = FP(0,2,-4,1)
# f1_train,_ = fp1.predict(x_train)
# f1_test,_ = fp1.predict(x_test)
# f2_train,_ = fp2.predict(x_train)
# f2_test,_ = fp2.predict(x_test)

# f_train = np.concatenate([f1_train, f2_train], axis = 1)
# f_test = np.concatenate([f1_test, f2_test], axis = 1)

# # Plot function overlayed with predicted
# fig = plt.figure(figsize=(16,9)); 
# ax = fig.add_subplot()
# ax.plot(x_test, fitxp['mmean'], 'green')
# ax.plot(x_test, f0_test, 'black')
# ax.plot(x_test, f_test[:,0], 'r')
# ax.plot(x_test, f_test[:,1], 'b')
# ax.scatter(x_train, y_train)
# ax.set_xlabel("x"); ax.set_ylabel("y")
# plt.show()


# # Plot weight functions with 95% intervals
# fig = plt.figure(figsize=(16,9)); 
# ax = fig.add_subplot()
# ax.plot(x_test, fitxw['wmean'][:,0], 'red')
# ax.plot(x_test, fitxw['wmean'][:,1], 'blue')
# ax.set_xlabel("x"); ax.set_ylabel("w(x)")
# plt.show()

# fitxp['mmean']
# fitxp['smean']
# fitxw['wmean']
