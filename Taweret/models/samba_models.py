###########################################################
# Models from the SAndbox for Mixing using Bayesian
# Analysis (SAMBA) package
# Author : Alexandra Semposki
# Original models : M. Honda, JHEP 12, 019 (2014).
# ##########################################################

# imports
from Taweret.utils.utils import normal_log_likelihood_elementwise as log_likelihood_elementwise_utils
from Taweret.core.base_model import BaseModel
import numpy as np
from scipy import special, integrate
import math as math
import sys
sys.path.append('../../Taweret')


__all__ = ['Loworder', 'Highorder', 'TrueModel', 'Data']


class Loworder(BaseModel):

    def __init__(self, order, error_model='informative'):
        """

        The SAMBA loworder series expansion function.
        This model has been previously calibrated.

        Parameters
        ----------
        order : int
            Truncation order of expansion
        error_model : str
            Error calculation method. Either 'informative' or
            'uninformative'

        Raises
        ------
        TypeError
            If the order is not an integer
        """

        if isinstance(order, int):
            self.order = order
        else:
            raise TypeError(f"order has to be an integer number: {order}")

        self.error_model = error_model
        self.prior = None

    def evaluate(self, input_values: np.array) -> np.array:
        """
        Evaluate the mean and standard deviation for
        given input values to the function

        Parameters
        ----------
        input_values : numpy 1darray
            coupling strength (g) values

        Returns:
        --------
        mean : numpy 1darray
            The mean of the model
        np.sqrt(var) : numpy 1darray
            The truncation error of the model
        """

        self.x = input_values

        # model function
        output = []

        low_c = np.empty([int(self.order) + 1])
        low_terms = np.empty([int(self.order) + 1])

        # if g is an array, execute here
        try:
            value = np.empty([len(self.x)])

            # loop over array in g
            for i in range(len(self.x)):

                # loop over orders
                for k in range(int(self.order) + 1):

                    if k % 2 == 0:
                        low_c[k] = np.sqrt(
                            2.0) * special.gamma(k + 0.5) * (-4.0)**(k // 2) / (math.factorial(k // 2))
                    else:
                        low_c[k] = 0

                    low_terms[k] = low_c[k] * self.x[i]**(k)

                value[i] = np.sum(low_terms)

            output.append(value)
            data = np.array(output, dtype=np.float64)

        # if g is a single value, execute here
        except BaseException:
            value = 0.0
            for k in range(int(self.order) + 1):

                if k % 2 == 0:
                    low_c[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * \
                        (-4.0)**(k // 2) / (math.factorial(k // 2))
                else:
                    low_c[k] = 0

                low_terms[k] = low_c[k] * self.x**(k)

            value = np.sum(low_terms)
            data = value

        # rename for clarity
        mean = data

        # uncertainties function
        # even order
        if self.order % 2 == 0:

            # find coefficients
            c = np.empty([int(self.order + 2)])

            # model 1 for even orders
            if self.error_model == 'uninformative':

                for k in range(int(self.order + 2)):

                    if k % 2 == 0:
                        c[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(
                            k // 2) / (math.factorial(k) * math.factorial(k // 2))
                    else:
                        c[k] = 0.0

                # rms value
                cbar = np.sqrt(np.sum((np.asarray(c))**2.0) /
                               (self.order // 2 + 1))

                # variance
                var1 = (cbar)**2.0 * (math.factorial(self.order + 2)
                                      )**2.0 * self.x**(2.0 * (self.order + 2))

            # model 2 for even orders
            elif self.error_model == 'informative':

                for k in range(int(self.order + 2)):

                    if k % 2 == 0:

                        # skip first coefficient
                        if k == 0:
                            c[k] = 0.0
                        else:
                            c[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(k // 2) / (
                                math.factorial(k // 2) * math.factorial(k // 2 - 1) * 4.0**(k))
                    else:
                        c[k] = 0.0

                # rms value
                cbar = np.sqrt(np.sum((np.asarray(c))**2.0) /
                               (self.order // 2 + 1))

                # variance
                var1 = (cbar)**2.0 * (math.factorial(self.order // 2)
                                      )**2.0 * (4.0 * self.x)**(2.0 * (self.order + 2))

        # odd order
        else:

            # find coefficients
            c = np.empty([int(self.order + 1)])

            # model 1 for odd orders
            if self.error_model == 'uninformative':

                for k in range(int(self.order + 1)):

                    if k % 2 == 0:
                        c[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(
                            k // 2) / (math.factorial(k) * math.factorial(k // 2))
                    else:
                        c[k] = 0.0

                # rms value
                cbar = np.sqrt(np.sum((np.asarray(c))**2.0) /
                               (self.order // 2 + 1))

                # variance
                var1 = (cbar)**2.0 * (math.factorial(self.order + 1)
                                      )**2.0 * self.x**(2.0 * (self.order + 1))

            # model 2 for odd orders
            elif self.error_model == 'informative':

                for k in range(int(self.order + 1)):

                    if k % 2 == 0:

                        # skip first coefficient
                        if k == 0:
                            c[k] = 0.0
                        else:
                            c[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(k // 2) / (
                                math.factorial(k // 2) * math.factorial(k // 2 - 1) * 4.0**(k))
                    else:
                        c[k] = 0.0

                # rms value
                cbar = np.sqrt(np.sum((np.asarray(c))**2.0) /
                               (self.order // 2 + 1))

                # variance
                var1 = (cbar)**2.0 * (math.factorial((self.order - 1) // 2)
                                      )**2.0 * (4.0 * self.x)**(2.0 * (self.order + 1))

        # rename for clarity
        var = var1

        return mean.flatten(), np.sqrt(var).flatten()

    def log_likelihood_elementwise(self, x_exp, y_exp, y_err, model_param):
        return log_likelihood_elementwise_utils(
            self, x_exp, y_exp, y_err, model_param)

    def set_prior(self):
        '''
        Set the prior on model parameters.
        Not needed for this model.
        '''
        return None


class Highorder(BaseModel):

    def __init__(self, order, error_model='informative'):
        """
        The SAMBA highorder series expansion function.

        Parameters
        ----------
        order : int
            Truncation order of expansion
        error_model : str
            Error calculation method. Either 'informative' or 'uninformative'

        Raises
        ------
        TypeError
            If the order is not an integer
        """

        if isinstance(order, int):
            self.order = order
        else:
            raise TypeError(f"order has to be an integer number: {order}")

        self.error_model = error_model
        self.prior = None

    def evaluate(self, input_values: np.array) -> np.array:
        """
        Evaluate the mean and standard deviation for given
        input values

        Parameters
        ----------
        input_values : numpy 1darray
            coupling strength (g) values

        Returns:
        --------
        mean : numpy 1darray
            The mean of the model
        np.sqrt(var) : numpy 1darray
            The truncation error of the model
        """

        order = self.order
        self.x = input_values

        # mean function
        output = []

        high_c = np.empty([int(order) + 1])
        high_terms = np.empty([int(order) + 1])

        # if g is an array, execute here
        try:
            value = np.empty([len(self.x)])

            # loop over array in g
            for i in range(len(self.x)):

                # loop over orders
                for k in range(int(order) + 1):

                    high_c[k] = special.gamma(
                        k / 2.0 + 0.25) * (-0.5)**k / (2.0 * math.factorial(k))

                    high_terms[k] = (high_c[k] * self.x[i] **
                                     (-k)) / np.sqrt(self.x[i])

                # sum the terms for each value of g
                value[i] = np.sum(high_terms)

            output.append(value)

            data = np.array(output, dtype=np.float64)

        # if g is a single value, execute here
        except BaseException:
            value = 0.0

            # loop over orders
            for k in range(int(order) + 1):

                high_c[k] = special.gamma(
                    k / 2.0 + 0.25) * (-0.5)**k / (2.0 * math.factorial(k))

                high_terms[k] = (high_c[k] * self.x**(-k)) / np.sqrt(self.x)

            # sum the terms for each value of g
            value = np.sum(high_terms)
            data = value

        # rename for clarity
        mean = data

        # uncertainties function
        # find coefficients
        d = np.zeros([int(self.order) + 1])

        # model 1
        if self.error_model == 'uninformative':

            for k in range(int(self.order) + 1):

                d[k] = special.gamma(k / 2.0 + 0.25) * (-0.5)**k * \
                    (math.factorial(k)) / (2.0 * math.factorial(k))

            # rms value (ignore first two coefficients in this model)
            dbar = np.sqrt(np.sum((np.asarray(d)[2:])**2.0) / (self.order - 1))

            # variance
            var2 = (dbar)**2.0 * (self.x)**(-1.0) * (math.factorial(self.order + 1)
                                                     )**(-2.0) * self.x**(-2.0 * self.order - 2)

        # model 2
        elif self.error_model == 'informative':

            for k in range(int(self.order) + 1):

                d[k] = special.gamma(k / 2.0 + 0.25) * special.gamma(
                    k / 2.0 + 1.0) * 4.0**(k) * (-0.5)**k / (2.0 * math.factorial(k))

            # rms value
            dbar = np.sqrt(np.sum((np.asarray(d))**2.0) / (self.order + 1))

            # variance
            var2 = (dbar)**2.0 * self.x**(-1.0) * (special.gamma((self.order + \
                    3) / 2.0))**(-2.0) * (4.0 * self.x)**(-2.0 * self.order - 2.0)

        # rename for clarity
        var = var2

        return mean.flatten(), np.sqrt(var).flatten()

    def log_likelihood_elementwise(self, x_exp, y_exp, y_err, model_param):
        return log_likelihood_elementwise_utils(
            self, x_exp, y_exp, y_err, model_param)

    def set_prior(self):
        '''
        Set the prior on the model parameters.
        Not needed for this model.
        '''
        return None


class TrueModel(BaseModel):

    def evaluate(self, input_values: np.array) -> np.array:
        """
        Evaluate the mean of the true model for given input values.

        Parameters:
        ----------
        input_values : numpy 1darray
            coupling strength (g) values

        Returns:
        --------
        mean : numpy 1darray
            The true model evaluated at each point of the
            given input space
        np.sqrt(var) : numpy 1darray
            The standard deviation of the true model. This
            will obviously be an array of zeros.
        """

        self.x = input_values

        # true model
        def function(x, g):
            return np.exp(-(x**2.0) / 2.0 - (g**2.0 * x**4.0))

        # initialization
        self.model = np.zeros([len(self.x)])

        # perform the integral for each g
        for i in range(len(self.x)):

            self.model[i], self.err = integrate.quad(
                function, -np.inf, np.inf, args=(self.x[i],))

        mean = self.model
        var = np.zeros(shape=mean.shape)

        return mean.flatten(), np.sqrt(var).flatten()

    def log_likelihood_elementwise(self, x_exp, y_exp, y_err):
        return log_likelihood_elementwise_utils(self, x_exp, y_exp, y_err)

    def set_prior(self):
        '''
        Set the prior on any model parameters.
        Not needed for this model.
        '''
        return None


class Data(BaseModel):    # --> check that this model is set up correctly

    def evaluate(self, input_values: np.array, error=0.01) -> np.array:
        """
        Evaluate the data and error for given input values

        Parameters:
        ----------
        input_values : numpy 1darray
            coupling strength (g) values for data generation
        error : float
            defines the relative error as a fraction between (0,1)

        Returns:
        --------
        data : numpy 1darray
            The array of data points
        sigma : numpy 1darray
            The errors on each data point
        """

        # call class for true model
        truemodel = TrueModel()

        # data generation input values
        x_data = input_values

        # adding data using the add_data function from SAMBA
        if error is None:
            raise ValueError(
                'Please enter a error in decimal form for the data set generation.')
        elif error < 0.0 or error > 1.0:
            raise ValueError('Error must be between 0.0 and 1.0.')

        # generate fake data
        data, _ = truemodel.evaluate(x_data)
        rand = np.random.RandomState()
        var = error * rand.randn(len(x_data))
        data = data * (1 + var)

        # calculate standard deviation
        sigma = error * data

        return data, sigma

    def log_likelihood_elementwise(self):
        '''
        Obtain the log likelihood for the model.
        Not needed for this model.
        '''
        return None

    def set_prior(self):
        '''
        Set the prior on any model parameters.
        Not needed for this model.
        '''
        return None
