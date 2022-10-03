import numpy as np


class coleman_model_1():

    def predict(self, input_values : np.array, model_param : np.array) -> np.array:
        """
        Predict the mean and error for given input values

        Parameters
        ----------
        input_values : numpy 1darray
            input parameter values
        model_param : numpy 1darray
            value of the model parameter
        """

        x = input_values.flatten()
        mean = np.zeros(len(x))
        var = 0.3*0.3*np.zeros(len(x))

        if len(model_param.flatten()) !=1 :
            raise TypeError('The model_param has to be single element numpy array')

        mean = 0.5 * (x + model_param.item()) - 2
        return mean, np.sqrt(var)

class coleman_model_2():

    def predict(self, input_values : np.array, model_param : np.array) -> np.array:
        """
        Predict the mean and error for given input values

        Parameters
        ----------
        input_values : numpy 1darray
            input parameter values
        model_param : numpy 1darray
            value of the model parameter
        """

        x = input_values.flatten()
        mean = np.zeros(len(x))
        var = 0.3*0.3*np.zeros(len(x))

        if len(model_param.flatten()) !=1 :
            raise TypeError('The model_param has to be single element numpy array')

        mean = -0.5 * (x - model_param.item()) + 3.7
        return mean, np.sqrt(var)      

class coleman_truth():

    def predict(self, input_values : np.array) -> np.array:
        """
        Predict the mean and error for given input values

        Parameters
        ----------
        input_values : numpy 1darray
            input parameter values
        """

        x = input_values.flatten()
        mean = np.zeros(len(x))
        var = 0.3* 0.3 * np.ones(len(x))

        mean = 2 - 0.1 * (x - 4)**2
        return mean, np.sqrt(var)