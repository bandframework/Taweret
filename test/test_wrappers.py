# ---------------------------------------------------------
# Testing Mixing wrapper
# ---------------------------------------------------------
import numpy as np
import Taweret as twr # Easy import of Taweret package, default imports controlled by __init__.py() 

#from Taweret.core.wrappers import mixing, IMPLEMENTED_MIXERS

# reloads, to delete later (not working as needed right now)
# import importlib
# importlib.reload(twr)
# importlib.reload(twr.core.base)
# importlib.reload(twr.mix.trees)
# from Taweret.core.base import BaseMixer

#----------------------------------------------------------
# Generate Example Data
#----------------------------------------------------------
import matplotlib.pyplot as plt

# Define Polynomial class to get predictions from the model f(x) = c*(x-a)^p + b
# Only used for mean predictions
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
# Line 1 of arguments = required arguments
# Lines 2 and 3 = method specific kwargs
m = twr.mixing(model_list, x_train, y_train, None, method = 'mixbart', 
            ntree = 10, k = 1, overallnu = 10, overallsd = 0.30, minnumbot = 1,
            local_openbt_path = "/home/johnyannotty/Documents/Open BT Project SRC")

# Train the model using one universal train() method
fit = m.train()

# Get predictions at a grid x_test using one universal predict() method
pred = m.predict(x_test, q_lower = 0.025, q_upper = 0.975)

# Get weights at a grid x_test using one universal weights() method
wts = m.weights(x_test, q_lower = 0.025, q_upper = 0.975)

# Get the mix_likelihood -- not implemented for BART yet so we can see the default output as defined in the base class
m.mix_likelihood(x_test)

# plot the weights using one universal plot_weights() methd
m.plot_weights(0)

# plot the predictions using one universal plot_prediction() methd
m.plot_prediction(0)

m.F_test
#------------------------------------------------
# Play with results -- delete later
pred['mmean']
wts['wmean']

col_list = ['red','blue']
fig = plt.figure(figsize=(6,5))  
for i in range(2):
    plt.plot(x_test, wts['wmean'][:,i], color = col_list[i])

plt.title("Posterior Weight Functions")
plt.xlabel("X") # Update Label
plt.ylabel("W(X)") # Update Label 
plt.grid(True, color='lightgrey')
plt.show()

col_list = ['red','blue','green','purple','orange']
xdim = 0
# Now plot the prediction -- need to improve this plot
fig = plt.figure(figsize=(6,5))  
plt.plot(m.X_test[:,xdim], pred['mmean'], color = 'black')
plt.plot(m.X_test[:,xdim], pred['m_lower'], color = 'black', linestyle = "dashed")
plt.plot(m.X_test[:,xdim], pred['m_upper'], color = 'black', linestyle = "dashed")
for i in range(2):
    plt.plot(m.X_test[:,xdim], m.F_test[:,i], color = col_list[i], linestyle = 'dotted')
plt.scatter(m.X_train[xdim,:] ,m.y_orig) # Recall X_train was transposed in the beginning 
plt.title("Posterior Mean Prediction")
plt.xlabel("X") # Update Label
plt.ylabel("F(X)") # Update Label 
plt.grid(True, color='lightgrey')
plt.show()
