# ---------------------------------------------------------
# Testing Mixing wrapper
# ---------------------------------------------------------
# Imports (make nicer once finished __init__.py() files in various folders)
import numpy as np
from Taweret.core.wrappers import mixing, IMPLEMENTED_MIXERS

# Data
import matplotlib.pyplot as plt
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

model_list = [FP(0,-2,4,1), FP(0,2,-4,1)]

# Training Data
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


# Try running the trees mixer
m = mixing(model_list, x_train, y_train, None, method = 'mixbart', ntree = 10, k = 1, local_openbt_path = "/home/johnyannotty/Documents/Open BT Project SRC")
fit = m.train()
pred = m.predict(x_test, q_lower = 0.025, q_upper = 0.975)
wts = m.weights(x_test, q_lower = 0.025, q_upper = 0.975)

m.mix_likelihood(x_test)

pred['mmean']
wts['wmean']