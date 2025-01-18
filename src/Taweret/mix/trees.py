# Name:
# trees.py
# Author: John Yannotty (yannotty.1@osu.edu)
# Start Date: 10/05/22
# Version: 1.0

import numpy as np
from openbtmixing import Openbtmix

from Taweret.core.base_mixer import BaseMixer



class Trees(BaseMixer):
    r'''
        Constructor for the Trees mixing class,
        which implements a mean-mixing strategy.
        The weight functions are modeled using
        Bayesian Additive Regression Trees (BART).
        Please read the installation page of the
        documentation to ensure the BART-BMM Ubuntu
        package is downloaded and installed.

        .. math::
            f_\dagger(x) = \sum_{k = 1}^K w_k(x)\;f_k(x)

        Example:
        --------

        .. code-block:: python

            # Initialize trees class
            mix = Trees(model_dict = model_dict)

            # Set prior information
            mix.set_prior(k=2.5,ntree=30,overallnu=5,
            overallsd=0.01,inform_prior=False)

            # Train the model
            fit = mix.train(X=x_train, y=y_train, ndpost = 10000,
            nadapt = 2000, nskip = 2000, adaptevery = 500, minnumbot = 4)

            # Get predictions and posterior weight functions.
            ppost, pmean, pci, pstd = mix.predict(X = x_test, ci = 0.95)
            wpost, wmean, wci, wstd = mix.predict_weights(X=x_test,ci = 0.95)

    '''

    def __init__(self, model_dict: dict, **kwargs):
        '''

        Parameters:
        -----------
        :param dict model_dict:
            Dictionary of models where each item is an
            instance of BaseModel.

        :param dict kwargs:
            Additional arguments to pass to the constructor.

        Returns:
        ---------
        :returns: None.
        '''
# Store model dictionary if all models are instances of BaseModel
        self.model_dict = model_dict
        self.nummodels = len(model_dict)
        self.obt = Openbtmix(**kwargs)

    def evaluate(self):
        '''
        Evaluate the mixed-model to get a point prediction.
        This method is not applicable to BART-based mixing.
        '''

        raise Exception("Not applicable for trees.")

    def evaluate_weights(self):
        '''
        Evaluate the weight functions to get a point prediction.
        This method is not applicable to BART-based mixing.
        '''
        raise Exception("Not applicable for trees.")

    @property
    def map(self):
        '''
        Return the map values for parameters in the model.
        This method is not applicable to BART-based mixing.
        '''
        return super().map

    @property
    def posterior(self):
        '''
        Returns the posterior distribution of the error standard deviation,
        which is learned during the training process.

        Parameters:
        ------------
        :param: None.

        Returns:
        ---------
        :returns: The posterior of the error standard deviation .
        :rtype: np.ndarray

        '''
        return self._posterior

    @property
    def prior(self):
        '''
        Returns a dictionary of the hyperparameter settings used in the
        various prior distributions.

        Parameters:
        -----------
        :param: None.

        Returns:
        --------
        :returns: A dictionary of the hyperparameters used in the model.
        :rtype: dict

        '''
        return self.obt.get_prior()

    def set_prior(
            self,
            ntree: int = 1,
            ntreeh: int = 1,
            k: float = 2,
            power: float = 2.0,
            base: float = 0.95,
            sighat: float = 1,
            nu: int = 10,
            inform_prior: bool = True):
        '''
        Sets the hyperparameters in the tree and terminal node priors. Also
        specifies if an informative or non-informative prior will be used
        when mixing EFTs.

        Parameters:
        -----------
        :param int ntree:
            The number of trees used in the sum-of-trees model for
            the weights.
        :param int ntreeh:
            The number of trees used in the product-of-trees model
            for the error standard deviation. Set to 1 for
            homoscedastic variance assumption.
        :param float k:
            The tuning parameter in the prior variance of the terminal node
            parameter prior. This is a value greater than zero.
        :param float power:
            The power parameter in the tree prior.
        :param float base:
            The base parameter in the tree prior.
        :param float overallsd:
            An initial estimate of the error standard deviation.
            This value is used to calibrate the scale parameter in
            variance prior.
        :param float overallnu:
            The shape parameter in the error variance prior.
        :param bool inform_prior:
            Controls if the informative or non-informative prior
            is used.
            Specify true for the informative prior.
        :param np.ndarray tauvec:
            A K-dimensional array (where K is the number of models)
            that contains the prior standard deviation of the terminal node
            parameter priors. This is used when specifying different
            priors for the different model weights.
        :param np.ndarray betavec:
            A K-dimensional array (where K is the number of models) that
            contains the prior mean of the terminal node
            parameter priors. This is used when specifying different
            priors for the different model weights.

        Returns:
        --------
        :returns: None.

        '''
        self.obt.set_prior(ntree,ntreeh,k,power,base,sighat,nu,inform_prior)

    def prior_predict(self):
        '''
        Return the prior predictive distribution of the mixed-model.
        This method is not applicable to BART-based mixing.
        '''
        raise Exception("Not applicable for trees at the moment.")

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs):
        '''
        Train the mixed-model using a set of observations y at inputs x.

        Parameters:
        -----------
        :param np.ndarray X: input parameter values of dimension (n x p).
        :param np.ndarray y: observed data at inputs X of dimension  (n x 1).
        :param dict kwargs: dictionary of arguments

        Returns:
        --------
        :returns: A dictionary which contains relevant information to the
            model such as values of tuning parameters.  The MCMC results are
            written to a text file and stored in a temporary directory as
            defined by the fpath key in the results dictionary.
        :rtype: dict
        '''
        # Cast data to arrays if not already and reshape if needed
        if isinstance(X, list):
            X = np.array(X)

        if len(X.shape) == 1:
            X = X.reshape(X.shape[0], 1)


        # Get predictions from the model set at X's
        fhat_list = []
        shat_list = []
        for m in list(self.model_dict.values()):
            # Get predictions from selected model
            fhat_col, shat_col = m.evaluate(X)

            # Append predictions to respective lists
            fhat_list.append(fhat_col)
            shat_list.append(shat_col)

        # Construct two matrices using concatenate
        f_matrix = np.concatenate(fhat_list, axis=1)
        s_matrix = np.concatenate(shat_list, axis=1)

        # Run the train command in openbtmixing
        res = self.obt.train(x_train = X, y_train = y, f_train = f_matrix, 
                             s_train = s_matrix, **kwargs)

        # Get predictions at training points -- more importanlty, 
        # get the posterior of sigma
        # ci level doesn't matter here, all we want is the posterior
        # using tc*2 just to get a small subset of data that 
        # won't break the array structures when reading in results
        res_sig = self.obt.predict(X[0:(self.obt.tc*2),], 
                                   f_matrix[0:(self.obt.tc*2),], ci=0.68)
        self._posterior = res_sig["sigma"]["draws"][:,0]

        return res

    def predict(self, X: np.ndarray, ci: float = 0.95):
        '''
        Obtain the posterior predictive distribution of the mixed-model
        at a set of inputs X.

        Parameters:
        -----------
        :param np.ndarray X: design matrix of testing inputs.
        :param float ci: credible interval width, must be a value
                within the interval (0,1).

        Returns:
        --------
        :returns: The posterior prediction draws and summaries.
        :rtype: np.ndarray, np.ndarray, np.ndarray, np.ndarray
        :return value: the posterior predictive distribution
                evaluated at the specified test points
        :return value: the posterior mean of the mixed-model
                at each input in X.
        :return value: the pointwise credible intervals at each input in X.
        :return value: the posterior standard deviation of the
                mixed-model at each input in X.
        '''

        # Set q_lower and q_upper
        alpha = (1 - ci)
        q_lower = alpha / 2
        q_upper = 1 - alpha / 2

        # Casting lists to arrays when needed
        if (isinstance(X, list)):
            X = np.array(X)
        if (len(X.shape) == 1):  # If shape is (n, ), change it to (n, 1):
            X = X.reshape(len(X), 1)

        # Get predictions from the model set at X's
        fhat_list = []
        shat_list = []
        for m in list(self.model_dict.values()):
            # Get predictions from selected model
            fhat_col, shat_col = m.evaluate(X)

            # Append predictions to respective lists
            fhat_list.append(fhat_col)
            shat_list.append(shat_col)

        # Construct F matrix using concatenate
        f_test = np.concatenate(fhat_list, axis=1)
        
        # Set control values
        self.p_test = X.shape[1]
        self.n_test = X.shape[0]

        self.q_lower = q_lower
        self.q_upper = q_upper

        # predict via openbtmixing grid
        res = self.obt.predict(X,f_test,ci)

        posterior = res["pred"]["draws"]
        post_mean = res["pred"]["mean"]
        post_sd = res["pred"]["sd"]
        post_credible_interval = [res["pred"]["lb"], res["pred"]["ub"]]

        return posterior, post_mean, post_credible_interval, post_sd

    def predict_weights(self, X: np.ndarray, ci: float = 0.95):
        '''
        Obtain posterior distribution of the weight functions at a set
        of inputs X.

        Parameters:
        -----------
        :param np.ndarray X: design matrix of testing inputs.
        :param float ci: credible interval width, must be a value
                within the interval (0,1).

        Returns:
        --------
        :returns: The posterior weight function draws and summaries.
        :rtype: np.ndarray, np.ndarray, np.ndarray, np.ndarray
        :return value: the posterior draws of the weight functions
                at each input in X.
        :return value: posterior mean of the weight functions at
                each input in X.
        :return value: pointwise credible intervals for
                the weight functions.
        :return value: posterior standard deviation of
                the weight functions at each input in X.
        '''

        # Set q_lower and q_upper
        alpha = (1 - ci)
        q_lower = alpha / 2
        q_upper = 1 - alpha / 2
       
        self.q_lower = q_lower
        self.q_upper = q_upper

        # predict via openbtmixing grid
        res = self.obt.predict_weights(X,ci)

        posterior = res["wts"]["draws"]
        post_mean = res["wts"]["mean"]
        post_sd = res["wts"]["sd"]
        post_credible_interval = [res["wts"]["lb"], res["wts"]["ub"]]

        return posterior, post_mean, post_credible_interval, post_sd