# This toy example is copied from Alexandra; SAMBA package
# : \pi^4 expansion for small and large g :
# Each model should have a predict and get parameters function:
# Predict function should output a mean and 1\sigma uncertainity (assuming the error is Gaussian distributed)
import numpy as np
import math
from scipy import special, integrate

class loworder():

    def __init__(self, order, error_model='informative'):
        if isinstance(order, int):
            self.order = order
        else:
            raise TypeError(f"order has to be an integer number: {order}")

        #assign error model 
        if error_model == 'uninformative':
            self.error_model = 1

        elif error_model == 'informative':
            self.error_model = 2
        else:
            raise ValueError("Please choose 'uninformative' or 'informative'.")

    def predict(self, input_values:np.array)->np.array:

        output = []
        g = input_values
        for order in [self.order]:
            low_c = np.empty([int(order)+1])
            low_terms = np.empty([int(order) + 1])

            #if g is an array, execute here
            try:
                value = np.empty([len(g)])
       
                #loop over array in g
                for i in range(len(g)):      

                    #loop over orders
                    for k in range(int(order)+1):

                        if k % 2 == 0:
                            low_c[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(k//2) / (math.factorial(k//2))
                        else:
                            low_c[k] = 0

                        low_terms[k] = low_c[k] * g[i]**(k) 

                    value[i] = np.sum(low_terms)

                output.append(value)
                data = np.array(output, dtype = np.float64)
            
            #if g is a single value, execute here
            except:
                value = 0.0
                for k in range(int(order)+1):

                    if k % 2 == 0:
                        low_c[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(k//2) / (math.factorial(k//2))
                    else:
                        low_c[k] = 0

                    low_terms[k] = low_c[k] * g**(k) 

                value = np.sum(low_terms)
                data = value

    # Uncertainty calculation

        #even order 
        if self.order % 2 == 0:
            loworder=self.order
            #find coefficients
            c = np.empty([int(loworder + 2)])

            #model 1 for even orders
            if self.error_model == 1:

                for k in range(int(loworder + 2)):

                    if k % 2 == 0:
                        c[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(k//2) / (math.factorial(k) * math.factorial(k//2))
                    else:
                        c[k] = 0.0

                #rms value
                cbar = np.sqrt(np.sum((np.asarray(c))**2.0) / (loworder//2 + 1))

                #variance 
                var1 = (cbar)**2.0 * (math.factorial(loworder + 2))**2.0 * g**(2.0*(loworder + 2))

            #model 2 for even orders
            elif self.error_model == 2:

                for k in range(int(loworder + 2)):

                    if k % 2 == 0:

                        #skip first coefficient
                        if k == 0:
                            c[k] = 0.0
                        else:
                            c[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(k//2) / (math.factorial(k//2) \
                                   * math.factorial(k//2 - 1) * 4.0**(k))
                    else:
                        c[k] = 0.0

                #rms value
                cbar = np.sqrt(np.sum((np.asarray(c))**2.0) / (loworder//2 + 1))

                #variance
                var1 = (cbar)**2.0 * (math.factorial(loworder//2))**2.0 * (4.0 * g)**(2.0*(loworder + 2))

        #odd order
        else:
            loworder=self.order
            #find coefficients
            c = np.empty([int(loworder + 1)])

            #model 1 for odd orders
            if self.error_model == 1:

                for k in range(int(loworder + 1)):

                    if k % 2 == 0:
                        c[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(k//2) / (math.factorial(k) * math.factorial(k//2))
                    else:
                        c[k] = 0.0

                #rms value
                cbar = np.sqrt(np.sum((np.asarray(c))**2.0) / (loworder//2 + 1))

                #variance
                var1 = (cbar)**2.0 * (math.factorial(loworder + 1))**2.0 * g**(2.0*(loworder + 1))

            #model 2 for odd orders
            elif self.error_model == 2:

                for k in range(int(loworder + 1)):

                    if k % 2 == 0:

                        #skip first coefficient
                        if k == 0:
                            c[k] = 0.0
                        else:
                            c[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(k//2) / (math.factorial(k//2) \
                                    * math.factorial(k//2 - 1) * 4.0**(k))
                    else:
                        c[k] = 0.0

                #rms value
                cbar = np.sqrt(np.sum((np.asarray(c))**2.0) / (loworder//2 + 1))

                #variance
                var1 = (cbar)**2.0 * (math.factorial((loworder-1)//2))**2.0 * (4.0 * g)**(2.0*(loworder + 1))
        return data, var1


class highorder():

    def __init__(self, order, error_model='informative'):
        if isinstance(order, int):
            self.order = order
        else:
            raise TypeError(f"order has to be an integer number: {order}")

        #assign error model 
        if error_model == 'uninformative':
            self.error_model = 1

        elif error_model == 'informative':
            self.error_model = 2
        else:
            raise ValueError("Please choose 'uninformative' or 'informative'.")

    def predict(self, input_values:np.array)->np.array:

        output = []
        g=input_values
        for order in [self.order]:
            high_c = np.empty([int(order) + 1])
            high_terms = np.empty([int(order) + 1])
            
            #if g is an array, execute here
            try:
                value = np.empty([len(g)])
        
                #loop over array in g
                for i in range(len(g)):

                    #loop over orders
                    for k in range(int(order)+1):

                        high_c[k] = special.gamma(k/2.0 + 0.25) * (-0.5)**k / (2.0 * math.factorial(k))

                        high_terms[k] = (high_c[k] * g[i]**(-k)) / np.sqrt(g[i])

                    #sum the terms for each value of g
                    value[i] = np.sum(high_terms)

                output.append(value)

                data = np.array(output, dtype = np.float64)
        
            #if g is a single value, execute here           
            except:
                value = 0.0

                #loop over orders
                for k in range(int(order)+1):

                    high_c[k] = special.gamma(k/2.0 + 0.25) * (-0.5)**k / (2.0 * math.factorial(k))

                    high_terms[k] = (high_c[k] * g**(-k)) / np.sqrt(g) 

                #sum the terms for each value of g
                value = np.sum(high_terms)
                data = value

    # Uncertainty calculation

        #find coefficients
        highorder=self.order
        d = np.zeros([int(highorder) + 1])

        #model 1
        if self.error_model == 1:

            for k in range(int(highorder) + 1):

                d[k] = special.gamma(k/2.0 + 0.25) * (-0.5)**k * (math.factorial(k)) / (2.0 * math.factorial(k))

            #rms value (ignore first two coefficients in this model)
            dbar = np.sqrt(np.sum((np.asarray(d)[2:])**2.0) / (highorder-1))

            #variance
            var2 = (dbar)**2.0 * (g)**(-1.0) * (math.factorial(highorder + 1))**(-2.0) \
                    * g**(-2.0*highorder - 2)

        #model 2
        elif self.error_model == 2:

            for k in range(int(highorder) + 1):

                d[k] = special.gamma(k/2.0 + 0.25) * special.gamma(k/2.0 + 1.0) * 4.0**(k) \
                       * (-0.5)**k / (2.0 * math.factorial(k))

            #rms value
            dbar = np.sqrt(np.sum((np.asarray(d))**2.0) / (highorder + 1))

            #variance
            var2 = (dbar)**2.0 * g**(-1.0) * (special.gamma((highorder + 3)/2.0))**(-2.0) \
                    * (4.0 * g)**(-2.0*highorder - 2.0)
        return data, var2
    
class true_model():

    def predict(self, input_values:np.array)->np.array:
        g=input_values
        #define a function for the integrand
        def function(x,g):
            return np.exp(-(x**2.0)/2.0 - (g**2.0 * x**4.0)) 
    
        #initialization
        output = np.zeros([len(g)])
    
        #perform the integral for each g
        for i in range(len(g)):
            output[i], _ = integrate.quad(function, -np.inf, np.inf, args=(g[i],))
        
        return output 


