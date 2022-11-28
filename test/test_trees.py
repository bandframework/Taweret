"""
Name: test_trees.py
Author: John Yannotty (yannotty.1@osu.edu)
Desc: Tests the tree mixing class, which is an interface for BARTMM 

Start Date: 11/07/22
Version: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from Taweret.mix.trees import Trees

import importlib
import Taweret.mix.trees
importlib.reload(Taweret.mix.trees)

# ---------------------------------------------------------
# Polynomial function for candidate models
# ---------------------------------------------------------
class FP:
    def __init__(self,a=0,b=0,c=1,p=1):
        self.a = a
        self.b = b
        self.c = c
        self.p = p
    
    def predict(self, x):
        if isinstance(x, list):
            x = np.array(x)
        m = self.c*(x-self.a)**self.p + self.b
        if len(m.shape) == 1:
            m = m.reshape(m.shape[0],1) 
        s = np.array([1]*x.shape[0]).reshape(m.shape[0],1)
        return m,s

model_dict = {'model1':FP(0,-2,4,1), 'model2':FP(0,2,-4,1)}

# ---------------------------------------------------------
# Generate Training Data
# ---------------------------------------------------------
n_train = 15
n_test = 100
s = 0.1

x_train = np.concatenate([np.array([0.01,0.1,0.25]),np.linspace(0.45,1.0, n_train-3)])
x_test = np.linspace(0.01, 1.0, n_test)

np.random.seed(1234567)
fp = FP(0.5,0,8,2)
f0_train,_ = fp.predict(x_train)
f0_test,_ = fp.predict(x_test)
y_train = f0_train + np.random.normal(0,s,n_train).reshape(n_train,1)

# ---------------------------------------------------------
# Test BART model mixing with polynomials
# ---------------------------------------------------------
# Define priors
prior_dict = {'k':1.25,'ntree':20, 'overallnu':5, 'overallsd':np.sqrt(0.1)}

# Mixing with the non-informative prior
mix = Trees(model_dict = model_dict, local_openbt_path = "/home/johnyannotty/Documents/openbt")

mix.set_prior(prior_dict = prior_dict)
mix.prior
mix.train(X=x_train, y=y_train, ndpost = 10000, nadapt = 2000, nskip = 1000, adaptevery = 500, minnumbot = 2)

post = mix.posterior
np.mean(post)
len(post)

# Get predictions
ppost, pmean, pci, pstd = mix.predict(X = x_test, ci = 0.95)
wpost, wmean, wci, wstd = mix.predict_weights(X = x_test, ci = 0.95)

# Plot results
mix.plot_weights(0)
mix.plot_prediction(0)
mix.plot_sigma()

# Get the model predictions (used for next plot)
fp1 = FP(0,-2,4,1)
fp2 = FP(0,2,-4,1)
f1_train,_ = fp1.predict(x_train)
f1_test,_ = fp1.predict(x_test)
f2_train,_ = fp2.predict(x_train)
f2_test,_ = fp2.predict(x_test)

# Plot true function overlayed with predicted
plt.figure(figsize=(16,9)); 
plt.plot(x_test, pmean, 'green')
plt.plot(x_test, f0_test, 'black')
plt.plot(x_test, f1_test[:,0], 'r')
plt.plot(x_test, f2_test[:,0], 'b')
plt.scatter(x_train, y_train)
plt.show()

