import numpy as np
from scipy.special import factorial
from Taweret.core.base_model import BaseModel

# Polynomial Class Functions
class polynomal_model(BaseModel):
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

# Taylor Expansions
class sin_exp(BaseModel):
    def __init__(self,k,x0):
        self.k = k
        self.x0 = x0
    
    def evaluate(self, x):
        # Check type of x
        if isinstance(x, list):
            x = np.array(x)
        
        # Get degree list for polynomial expansion
        deg = np.linspace(0,self.k,self.k+1)
        
        # Get every 4th term (derivative repaets every 4 terms)
        h0 = deg[deg%4 == 0]
        h1 = deg[deg%4 == 1]
        h2 = deg[deg%4 == 2]
        h3 = deg[deg%4 == 3]
        
        # Compute taylor series:
        xc = x - self.x0
        xc0 = xc.repeat(h0.shape[0]).reshape(x.shape[0],h0.shape[0])
        ts = np.sum(np.sin(self.x0)*np.power(xc0,h0)/factorial(h0), axis = 1)
        
        xc1 = xc.repeat(h1.shape[0]).reshape(x.shape[0],h1.shape[0])
        ts = ts + np.sum(np.cos(self.x0)*np.power(xc1,h1)/factorial(h1), axis = 1)
        
        xc2 = xc.repeat(h2.shape[0]).reshape(x.shape[0],h2.shape[0])
        ts = ts + np.sum(-np.sin(self.x0)*np.power(xc2,h2)/factorial(h2), axis = 1)

        xc3 = xc.repeat(h3.shape[0]).reshape(x.shape[0],h3.shape[0])
        ts = ts + np.sum(-np.cos(self.x0)*np.power(xc3,h3)/factorial(h3), axis = 1)
        
        if len(ts.shape) == 1:
            ts = ts.reshape(ts.shape[0],1) 
        
        s = np.array([1]*x.shape[0]).reshape(ts.shape[0],1)
        return ts,s

    def set_prior(self):
        return super().set_prior()

    def log_likelihood_elementwise(self):
        return super().log_likelihood_elementwise()



class cos_exp(BaseModel):
    def __init__(self,k,x0):
        self.k = k
        self.x0 = x0
    
    def evaluate(self, x):
        # Check type of x
        if isinstance(x, list):
            x = np.array(x)
        
        # Get degree list for polynomial expansion
        deg = np.linspace(0,self.k,self.k+1)
        
        # Get every 4th term (derivative repaets every 4 terms)
        h0 = deg[deg%4 == 0]
        h1 = deg[deg%4 == 1]
        h2 = deg[deg%4 == 2]
        h3 = deg[deg%4 == 3]
        
        # Compute taylor series:
        xc = x - self.x0
        xc0 = xc.repeat(h0.shape[0]).reshape(x.shape[0],h0.shape[0])
        ts = np.sum(np.cos(self.x0)*np.power(xc0,h0)/factorial(h0), axis = 1)
        
        xc1 = xc.repeat(h1.shape[0]).reshape(x.shape[0],h1.shape[0])
        ts = ts + np.sum(-np.sin(self.x0)*np.power(xc1,h1)/factorial(h1), axis = 1)
        
        xc2 = xc.repeat(h2.shape[0]).reshape(x.shape[0],h2.shape[0])
        ts = ts + np.sum(-np.cos(self.x0)*np.power(xc2,h2)/factorial(h2), axis = 1)

        xc3 = xc.repeat(h3.shape[0]).reshape(x.shape[0],h3.shape[0])
        ts = ts + np.sum(np.sin(self.x0)*np.power(xc3,h3)/factorial(h3), axis = 1)
        
        if len(ts.shape) == 1:
            ts = ts.reshape(ts.shape[0],1) 
        
        s = np.array([1]*x.shape[0]).reshape(ts.shape[0],1)
        return ts,s

    def set_prior(self):
        return super().set_prior()

    def log_likelihood_elementwise(self):
        return super().log_likelihood_elementwise()


class sin_cos_exp(BaseModel):
    def __init__(self,ks,kc,xs,xc):
        self.ks = ks
        self.xs = xs

        self.kc = kc
        self.xc = xc
    
    def evaluate(self, x):
        # Check type of x
        if isinstance(x, list):
            x = np.array(x)
        
        # Sine Part
        # Get degree list for polynomial expansion
        deg = np.linspace(0,self.ks,self.ks+1)
        
        # Get every 4th term (derivative repaets every 4 terms)
        h0 = deg[deg%4 == 0]
        h1 = deg[deg%4 == 1]
        h2 = deg[deg%4 == 2]
        h3 = deg[deg%4 == 3]
        
        # Compute taylor series:
        xctr = x.transpose()[0] - self.xs
        xc0 = xctr.repeat(h0.shape[0]).reshape(x.shape[0],h0.shape[0])
        tss = np.sum(np.sin(self.xs)*np.power(xc0,h0)/factorial(h0), axis = 1)
        
        xc1 = xctr.repeat(h1.shape[0]).reshape(x.shape[0],h1.shape[0])
        tss = tss + np.sum(np.cos(self.xs)*np.power(xc1,h1)/factorial(h1), axis = 1)
        
        xc2 = xctr.repeat(h2.shape[0]).reshape(x.shape[0],h2.shape[0])
        tss = tss + np.sum(-np.sin(self.xs)*np.power(xc2,h2)/factorial(h2), axis = 1)

        xc3 = xctr.repeat(h3.shape[0]).reshape(x.shape[0],h3.shape[0])
        tss = tss + np.sum(-np.cos(self.xs)*np.power(xc3,h3)/factorial(h3), axis = 1)


        # Cosine Part
        # Get degree list for polynomial expansion
        deg = np.linspace(0,self.kc,self.kc+1)
        
        # Get every 4th term (derivative repaets every 4 terms)
        h0 = deg[deg%4 == 0]
        h1 = deg[deg%4 == 1]
        h2 = deg[deg%4 == 2]
        h3 = deg[deg%4 == 3]
        
        # Compute taylor series:
        xctr = x.transpose()[1] - self.xc
        xc0 = xctr.repeat(h0.shape[0]).reshape(x.shape[0],h0.shape[0])
        tsc = np.sum(np.cos(self.xc)*np.power(xc0,h0)/factorial(h0), axis = 1)
        
        xc1 = xctr.repeat(h1.shape[0]).reshape(x.shape[0],h1.shape[0])
        tsc = tsc + np.sum(-np.sin(self.xc)*np.power(xc1,h1)/factorial(h1), axis = 1)
        
        xc2 = xctr.repeat(h2.shape[0]).reshape(x.shape[0],h2.shape[0])
        tsc = tsc + np.sum(-np.cos(self.xc)*np.power(xc2,h2)/factorial(h2), axis = 1)

        xc3 = xctr.repeat(h3.shape[0]).reshape(x.shape[0],h3.shape[0])
        tsc = tsc + np.sum(np.sin(self.xc)*np.power(xc3,h3)/factorial(h3), axis = 1)
        
        # Add the sine and cosine parts
        ts = tss + tsc

        if len(ts.shape) == 1:
            ts = ts.reshape(ts.shape[0],1) 
        
        s = np.array([1]*x.shape[0]).reshape(ts.shape[0],1)
        return ts,s

    def set_prior(self):
        return super().set_prior()

    def log_likelihood_elementwise(self):
        return super().log_likelihood_elementwise()
