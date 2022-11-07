import numpy as np
from scipy import special, integrate 
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from cycler import cycler 

__all__ = ['Models', 'Uncertainties']


class Models():

    def __init__(self, loworder, highorder):

        '''
        The class containing the expansion models from Honda's paper
        and the means to plot them. 

        :Example:
            Models(loworder=np.array([2]), highorder=np.array([5]))

        Parameters:
        -----------
        loworder : numpy.ndarray, int, float
            The value of N_s to be used to truncate the small-g expansion.
            Can be an array of multiple values or one. 

        highorder : numpy.ndarray, int, float
            The value of N_l to be used to truncate the large-g expansion.
            Can be an array of multiple values or one. 

        Returns:
        --------
        None.
        '''

        #check type and assign to class variable
        if isinstance(loworder, float) == True or isinstance(loworder, int) == True:
            loworder = np.array([loworder])

        #check type and assign to class variable
        if isinstance(highorder, float) == True or isinstance(highorder, int) == True:
            highorder = np.array([highorder])

        self.loworder = loworder
        self.highorder = highorder 

        return None 


    def low_g(self, g):
        
        '''
        A function to calculate the small-g divergent asymptotic expansion for a given range in the coupling 
        constant, g.
        
        :Example:
            Models.low_g(g=np.linspace(0.0, 0.5, 20))
            
        Parameters:
        -----------
        g : linspace
            The linspace of the coupling constant for this calculation. 
           
        Returns:
        --------
        output : numpy.ndarray
            The array of values of the expansion in small-g at each point in g_true space, for each value of 
            loworder (highest power the expansion reaches).
        '''

        output = []
        
        for order in self.loworder:
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

        return data

        
    def high_g(self, g):
        
        '''
        A function to calculate the large-g convergent Taylor expansion for a given range in the coupling 
        constant, g.
        
        :Example:
            Models.high_g(g=np.linspace(1e-6,1.0,100))
            
        Parameters:
        -----------
        g : linspace
            The linspace of the coupling constant for this calculation.
            
        Returns
        -------
        output : numpy.ndarray        
            The array of values of the expansion at large-g at each point in g_true space, for each value of highorder
            (highest power the expansion reaches).
        '''

        output = []
        
        for order in self.highorder:
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
                
        return data 


    def true_model(self, g):
        
        '''
        The true model of the zero-dimensional phi^4 theory partition function using an input linspace.
        
        :Example:
            Models.true_model(g=np.linspace(0.0, 0.5, 100))
            
        Parameters:
        -----------
        g : linspace
            The linspace for g desired to calculate the true model. This can be the g_true linspace, g_data
            linspace, or another linspace of the user's choosing. 
            
        Returns:
        -------
        model : numpy.ndarray        
            The model calculated at each point in g space. 
        '''
    
        #define a function for the integrand
        def function(x,g):
            return np.exp(-(x**2.0)/2.0 - (g**2.0 * x**4.0)) 
    
        #initialization
        self.model = np.zeros([len(g)])
    
        #perform the integral for each g
        for i in range(len(g)):
            
            self.model[i], self.err = integrate.quad(function, -np.inf, np.inf, args=(g[i],))
        
        return self.model 
   

    def plot_models(self, g):
        
        '''
        A plotting function to produce a figure of the model expansions calculated in Models.low_g and Models.high_g, 
        and including the true model calculated using Mixing.true_model.
        
        :Example:
            Mixing.plot_models(g=np.linspace(0.0, 0.5, 100))
            
        Parameters:
        -----------
        g : linspace
            The linspace in on which the models will be plotted here. 

        Returns
        -------
        None.
        
        '''
        
        #set up the plot
        fig = plt.figure(figsize=(8,6), dpi=600)
        ax = plt.axes()
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.locator_params(nbins=8)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel('g', fontsize=22)
        ax.set_ylabel('F(g)', fontsize=22)

        #x and y limits
        if min(g) == 1e-6:
            ax.set_xlim(0.0, max(g))
        else:
            ax.set_xlim(min(g), max(g))
        ax.set_ylim(1.2,3.2)
        ax.set_yticks([1.2, 1.6, 2.0, 2.4, 2.8, 3.2])
  
        #plot the true model 
        ax.plot(g, self.true_model(g), 'k', label='True model')
        
        #add linestyle cycler
        linestyle_cycler = cycler(linestyle=['dashed', 'dotted', 'dashdot', 'dashed', 'dotted', 'dashdot'])
        ax.set_prop_cycle(linestyle_cycler)
                
        #for each large-g order, calculate and plot
        for i,j in zip(range(len(self.loworder)), self.loworder):
            ax.plot(g, self.low_g(g)[i,:], color='r', label=r'$f_s$ ($N_s$ = {})'.format(j))

        #for each large-g order, calculate and plot
        for i,j in zip(range(len(self.highorder)), self.highorder):
            ax.plot(g, self.high_g(g)[i,:], color='b', label=r'$f_l$ ($N_l$ = {})'.format(j))
                    
        ax.legend(fontsize=18, loc='upper right')
        plt.show()

        #save figure option
        # response = input('Would you like to save this figure? (yes/no)')

        # if response == 'yes':
        #     name = input('Enter a file name (include .jpg, .png, etc.)')
        #     fig.savefig(name, bbox_inches='tight')

        return None
        
         
    def residuals(self):
        
        '''
        A calculation and plot of the residuals of the model expansions vs the true model values at each point in g.
        g is set internally for this plot, as the plot must be shown in loglog format to see the power law of the
        residuals. 
        
        :Example:
            Mixing.residuals()
            
        Parameters:
        -----------
        None.
                  
        Returns:
        --------
        None. 
        
        '''
        
        #set up the plot
        fig = plt.figure(figsize=(8,6), dpi=600)
        ax = plt.axes()
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel('g', fontsize=22)
        ax.set_ylabel('Residual', fontsize=22)
        ax.set_title('F(g): residuals', fontsize=22)
        ax.set_xlim(1e-2, 10.)
        ax.set_ylim(1e-6, 1e17)

        #set range for g
        g_ext = np.logspace(-6., 6., 800)
        
        #set up marker cycler
        marker_cycler = cycler(marker=['.', '*', '+', '.', '*', '+'])
        ax.set_prop_cycle(marker_cycler)

        #calculate true model
        value_true = self.true_model(g_ext)
        
        #for each small-g order, plot
        valuelow = np.zeros([len(self.loworder), len(g_ext)])
        residlow = np.zeros([len(self.loworder), len(g_ext)])

        for i,j in zip(range(len(self.loworder)), self.loworder):
            valuelow[i,:] = self.low_g(g_ext)[i]
            residlow[i,:] = (valuelow[i,:] - value_true)/value_true
            ax.loglog(g_ext, abs(residlow[i,:]), 'r', linestyle="None", label=r"$F_s({})$".format(j))

        #for each large-g order, plot
        valuehi = np.zeros([len(self.highorder), len(g_ext)])
        residhi = np.zeros([len(self.highorder), len(g_ext)])

        for i,j in zip(range(len(self.highorder)), self.highorder):
            valuehi[i,:] = self.high_g(g_ext)[i]
            residhi[i,:] = (valuehi[i,:] - value_true)/value_true
            ax.loglog(g_ext, abs(residhi[i,:]), 'b', linestyle="None", label=r"$F_l({})$".format(j))
        
        ax.legend(fontsize=18)
        plt.show()


class Uncertainties:


    def __init__(self, error_model='informative'):

        '''
        An accompanying class to Models() that possesses the truncation error models
        that are included as variances with the small-g and large-g expansions. 

        :Example:
            Uncertainties()

        Parameters:
        -----------
        error_model : str
            The name of the error model to use in the calculation. Options are
            'uninformative' and 'informative'. Default is 'informative'.

        Returns:
        --------
        None.
        '''

        #assign error model 
        if error_model == 'uninformative':
           self.error_model = 1

        elif error_model == 'informative':
            self.error_model = 2

        else:
            raise ValueError("Please choose 'uninformative' or 'informative'.")

        return None


    def variance_low(self, g, loworder):


        '''
        A function to calculate the variance corresponding to the small-g expansion model.

        :Example:
            Bivariate.variance_low(g=np.linspace(1e-6, 0.5, 100), loworder=5)

        Parameters:
        -----------
        g : numpy.linspace
            The linspace over which this calculation is performed.

        loworder : int
            The order to which we know our expansion model. Must be passed one at a time if
            more than one model is to be calculated.

        Returns:
        --------
        var1 : numpy.ndarray
            The array of variance values corresponding to each value in the linspace of g. 
        '''

        #even order 
        if loworder % 2 == 0:
            
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

        return var1


    def variance_high(self, g, highorder):

        '''
        A function to calculate the variance corresponding to the large-g expansion model.

        :Example:
            Bivariate.variance_low(g=np.linspace(1e-6, 0.5, 100), highorder=23)

        Parameters:
        -----------
        g : numpy.linspace
            The linspace over which this calculation is performed.

        highorder : int
            The order to which we know our expansion model. This must be a single value.
            
        Returns:
        --------
        var2 : numpy.ndarray
            The array of variance values corresponding to each value in the linspace of g. 
        '''

        #find coefficients
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

        return var2

