"""
Name: test_trees.py
Author: John Yannotty (yannotty.1@osu.edu)
Desc: Tests the tree mixing class, which is an interface for BARTMM 

Start Date: 11/07/22
Version: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from Taweret.core.base_model import BaseModel
from Taweret.mix.trees import Trees

import importlib
import Taweret.mix.trees

importlib.reload(Taweret.mix.trees)
importlib.reload(Taweret.core.setup)

from Taweret.core.base_model import BaseModel


# ---------------------------------------------------------
# Polynomial function for candidate models
# ---------------------------------------------------------
class FP(BaseModel):
    def __init__(self,a=0,b=0,c=1,p=1):
        self.a = a
        self.b = b
        self.c = c
        self.p = p
    
    def evaluate(self, x):
        if isinstance(x, list):
            x = np.array(x)
        m = self.c*(x-self.a)**self.p + self.b
        if len(m.shape) == 1:
            m = m.reshape(m.shape[0],1) 
        s = np.array([1]*x.shape[0]).reshape(m.shape[0],1)
        return m,s

    def set_prior(self):
        return super().set_prior()

    def log_likelihood_elementwise(self):
        return super().log_likelihood_elementwise()


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
f0_train,_ = fp.evaluate(x_train)
f0_test,_ = fp.evaluate(x_test)
y_train = f0_train + np.random.normal(0,s,n_train).reshape(n_train,1)

# ---------------------------------------------------------
# Test BART model mixing with polynomials
# ---------------------------------------------------------
# Define priors
#prior_dict = {'k':1.25,'ntree':20, 'overallnu':5, 'overallsd':np.sqrt(0.1)}

# Mixing with the non-informative prior
mix = Trees(model_dict = model_dict, local_openbt_path = "/home/johnyannotty/Documents/openbt/src")
#mix = Trees(model_dict = model_dict)

mix.set_prior(k=1.25,ntree=2,overallnu=5,overallsd=np.sqrt(0.1),inform_prior=False)
mix.prior
fit = mix.train(X=x_train, y=y_train, ndpost = 10, nadapt = 2, nskip = 1, adaptevery = 500, minnumbot = 2)

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


# ---------------------------------------------------------
# Test BART model mixing with SAMBA
# ---------------------------------------------------------
import sys
import os

from subpackages.SAMBA import samba
import numpy as np
import matplotlib.pyplot as plt
from Taweret.mix.trees import Trees

samba.__annotations__
import Taweret as tw

sys.path.append("/home/johnyannotty/Taweret/subpackages/SAMBA")
from Taweret.models import samba_models as toy_models

m1 = toy_models.loworder(2, 'uninformative')
m2 = toy_models.highorder(4, 'uninformative')
f0 = toy_models.true_model()

x_train = np.linspace(0.03,0.5,20)
x_test = np.linspace(0.03,0.5,300)

y_train = f0.evaluate(x_train) + np.random.normal(loc = 0, scale = 0.005, size = x_train.size())

model_dict = {'model1':m1, 'model2':m2}


try:
    from subpackages.SAMBA.samba import models   # assuming you have SAMBA in your Taweret top directory 
    from subpackages.SAMBA.samba import mixing

except Exception as e:
    print(e)
    print('''To use the SAMBA toy models, SAMBA package needed to be installed first''')
    print('Then use `sys.path.append("path_to_local_SAMBA_instalation")` in your code before calling \
SAMBA models')

# ---------------------------------------------------------
# Test BART model mixing with 2D
# ---------------------------------------------------------
import sys
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import importlib

# Taweret imports
import Taweret.models.polynomial_models 
from Taweret.models.polynomial_models import sin_exp, cos_exp, sin_cos_exp
from Taweret.mix.trees import Trees

## Functions for design points 
def grid_2d_design(n1,n2,n, xmin = [-1,-1], xmax = [1,1]):
  # Generate n uniform rvs
  ux = np.random.uniform(0,1,n)
  uy = np.random.uniform(0,1,n)

  # Dimensions for each rectangle
  x1_len = (xmax[0] - xmin[0])/n1
  x2_len = (xmax[1] - xmin[1])/n2
  xgrid = [[x, y] for x in range(n1) for y in range(n2)]
  xgrid = np.array(xgrid).transpose()

  # Get points
  x1 = ux*x1_len + x1_len*xgrid[0] + xmin[0]
  x2 = uy*x2_len + x2_len*xgrid[1] + xmin[1]

  # Join data
  xdata = np.array([x1,x2]).transpose()
  return xdata


# Test

# Generate Data
n_train = 80
x_train = grid_2d_design(10,8,80,[-np.pi,-np.pi],[np.pi,np.pi])
f0_train = np.sin(x_train.transpose()[0]) + np.cos(x_train.transpose()[1])
y_train = f0_train + np.random.normal(0,0.1,n_train)


# Plot the surfaces
#x1_test = np.outer(np.linspace(-np.pi, np.pi, 50), np.ones(50))
#x2_test = x1_test.copy().transpose()
x1_test = np.outer(np.linspace(-3.2371, 3.1225, 50), np.ones(50))
x2_test = x1_test.copy().transpose()
f0_test = (np.sin(x1_test) + np.cos(x2_test))
x_test = np.array([x2_test.reshape(x2_test.size,),x1_test.reshape(x1_test.size,)]).transpose()

# Define color map
cmap = plt.get_cmap('hot')

# Creating figure
fig = plt.figure(figsize =(14, 9))
ax = plt.axes(projection ='3d')
 
# Creating plot
ax.plot_surface(x1_test, x2_test, f0_test, cmap = cmap)
plt.title("True System", size = 20)
plt.xlabel("x1", size = 14)
plt.ylabel("x2", size = 14)

# show plot
plt.show()


h1 = sin_cos_exp(7,10,np.pi,np.pi)
h2 = sin_cos_exp(13,6,-np.pi,-np.pi)
model_dict = {'model1':h1, 'model2':h2}
h1.evaluate(np.array([[np.pi,np.pi],[-np.pi,0]]))[0]
h1.evaluate(x_train)[0]
h2.evaluate(x_train)[0]
h1_out = h1.evaluate(x_test)[0]
h2_out = h2.evaluate(x_test)[0]
h1_out[:5]
h2_out[:5]
x_test
x1_test.shape

# Fit the BMM Model
mix = Trees(model_dict = model_dict, local_openbt_path = "/home/johnyannotty/Documents/openbt/src")

mix.set_prior(k=1.25,ntree=30,overallnu=5,overallsd=np.sqrt(0.15),inform_prior=False)
mix.prior
fit = mix.train(X=x_train, y=y_train, ndpost = 10000, nadapt = 2000, nskip = 2000, adaptevery = 500, minnumbot = 4)

# Get predictions
ppost, pmean, pci, pstd = mix.predict(X = x_test, ci = 0.95)
wpost, wmean, wci, wstd = mix.predict_weights(X = x_test, ci = 0.95)

# Plot
fig = plt.figure(figsize =(14, 9))
ax = plt.axes(projection ='3d')
 
# Creating plot
ax.plot_surface(x1_test, x2_test, pmean.reshape(x2_test.shape), cmap = cmap)
plt.title("Predicted System", size = 20)
plt.xlabel("x1", size = 14)
plt.ylabel("x2", size = 14)

# show plot
plt.show()