import numpy as np
import math
from ..utils.utils import log_likelihood_elementwise, mixture_function, eps
import matplotlib.pyplot as plt
from ..core import base_mixer, base_model
import bilby
from ..sampler.likelihood_wrappers import likelihood_wrapper_for_bilby
import os

class BivariateLinear(base_mixer):

    '''
    Local linear mixing of two models.

    '''

    def __init__(self, models_dic, method='sigmoid',
                nargs_model_dic=[], n_mix=0):
        '''
        Parameters
        ----------
        models_dic : dictionary {'name1' : model1, 'name2' : model2}
            Two models to mix, each must be derived from the base_model.
        method : str
            mixing function
        nargs_model_dic : dictionary {'name1' : N_model1, 'name2' : N_model2}
            number of free parameters in each model
        n_mix : int
            number of free parameters in the mixing funtion
        
        '''
        # check if more than two models are trying to be mixed
        if len(models_dic)!==2:
            raise Exception('Bivariate linear mixing requires only two models.\
                            Please look at the other mixing methods in Taweret \
                            for multi modal mixing')
        # check that lengths of lists are compatible
        # if len(models_dic) != len(nargs_model_dic) and len(nargs_model_dic) != 0:
        #     raise Exception("""in linear_mix.__init__: len(nargs_for_each_model) = 0 for
        #                     learning weights only 
        #                     otherwise len(nargs_for_each_model) = len(models_dic)""")

        #check for predict method in the models
        for i, model in enumerate(models_dic.items()):
            try:
                issubclass(model, base_model)
            except AttributeError:
                print(f'model {models_dic.keys()[i]} is not derived from core.base_model class')
            else:
                continue
            
        self.models_dic = models_dic

        #check if the mixing method exist
        if method not in ['step', 'sigmoid', 'cdf', 'switchcos']:
            raise Exception('Specefied mixing function is not found')

        self.method = method
        self.nargs_model_dic = nargs_model_dic
        self.n_mix = n_mix
        self.model_was_trained=False # Flag to know if the model was trained or not
        
        self.map=None
        self.prior=None
        self.posterior=None

    def evaluate(self, mixture_params, x , model_params=[]):

        '''
        Evaluvate the mixed model for given parameters at input values x

        Parameters
        ----------
        mixture_params : np.1darray
            parameter values that fix the shape of mixing function
        x : np.1daray
            input parameter values array
        model_params: list[np.1darray]
            list of parameter values for each model


        Returns
        ---------
        evaluation : np.1darray
            the evaluation of the mixed model at input values x
        
        '''

        weights = mixture_function(self.method, x, mixture_params)
        try:
            model_1_out,_ = self.models_dic.items()[0].evaluvate(x,model_params[0])
        except:
            model_1_out,_ = self.models_dic.items()[0].evaluvate(x)

        try:
            model_2_out,_ = self.models_dic.items()[1].evaluvate(x,model_params[1])
        except:
            model_2_out,_ = self.models_dic.items()[1].evaluvate(x)
        
        return weights[0]*model_1_out + weights[2]*model_2_out
    
    
    def evaluate_weights(self, mixture_params, x):
        '''
        return the mixing function value at the input parameter values x

        Parameters
        ----------
        mixture_params : np.1darray
            parameter values that fix the shape of mixing function
        x : np.1darray
            input parameter values
        
        Returns
        -------
        weights : list[np.1darray, np.1darray]
            weights for model 1 and model 2 at input values x

        '''

        return mixture_function(self.method, x, mixture_params)

    def map(self):
        pass

    def predict(self, x, CI=[5,95], samples=None):
        '''
        Evaluate posterior to make prediction at test points x.

        Parameters
        ----------
        x : np.1darray
            input parameter values
        CI : list
            confidence intervals
        samples: np.ndarray
            If samples are given use that instead of posterior\
                for predictions. 

        Returns:
        --------
        evaluated_posterior : np.ndarray
            array of posterior predictive distribution evaluated at provided
            test points
        mean : np.ndarray
            average mixed model value at each provided test points
        credible_intervals : np.ndarray
            intervals corresponding for 60%, 90% credible intervals
        std_dev : np.ndarray
            sample standard deviation of mixed model output at provided test
            points
        '''

        if self.model_was_trained==False and samples is None:
            raise Exception('Posterior is not available to make predictions\n\
                            train the model before predicting')
        pos_predictions = []
        
        if samples is not None:
            print("using provided samples instead of posterior")
            posterior = sample
        else:
            posterior = self.posterior.values
        for sample in posterior:
            sample = np.array(sample).flatten()

            mixture_param = sample[0:self.n_mix]
            model_params = []
            for i in range(0,2):
                model_params.append(sample[self.n_mix:self.n_mix+self.nargs_model_dic.items()[i]])

            value = self.evaluate(mixture_param,x, model_params)
            pos_predictions.append(value)
        pos_predictions = np.array(pos_predictions).reshape(-1,len(x))

        CIs = np.percentile(pos_predictions,CI, axis=0)

        mean = np.mean(pos_predictions, axis=0)

        std_dev = np.std(pos_predictions, axis=0)

        return pos_predictions, mean, CIs, std_dev

    def predict_weights(self, x, CI=[5,95], samples=None):

        '''
        Calculate posterior predictive distribution for first model weights

        Parameters
        ----------
        x : np.1darray
            input parameter values
        CI : list
            confidence intervals
        samples: np.ndarray
            If samples are given use that instead of posterior\
                for predictions. 
        Returns:
        --------
        posterior_weights : np.ndarray
            array of posterior predictive distribution of weights
        mean : np.ndarray
            average mixed model value at each provided test points
        credible_intervals : np.ndarray
            intervals corresponding for 60%, 90% credibl e intervals
        std_dev : np.ndarray
            sample standard deviation of mixed model output at provided test
            points
        '''

        if self.model_was_trained==False and samples is None:
            raise Exception('Posterior is not available to make predictions\n\
                            train the model before predicting')
        pos_predictions = []

        if samples is not None:
            print("using provided samples instead of posterior")
            posterior = sample
        else:
            posterior = self.posterior.values
        for sample in posterior:
            sample = np.array(sample).flatten()

            mixture_param = sample[0:self.n_mix]

            value = self.evaluate_weights(mixture_param,x)
            pos_predictions.append(value)
        pos_predictions = np.array(pos_predictions).reshape(-1,len(x))

        CIs = np.percentile(pos_predictions,CI, axis=0)

        mean = np.mean(pos_predictions, axis=0)

        std_dev = np.std(pos_predictions, axis=0)

        return pos_predictions, mean, CIs, std_dev


    def prior_predict(self, x, CI=[5,95], n_sample=10000):
        '''
        Evaluate prior to make prediction at test points x.

        Parameters
        ----------
        x : np.1darray
            input parameter values
        CI : list
            confidence intervals
        n_samples : int
            number of samples to evaluvate prior_prediction

        Returns:
        --------
        evaluated_prior : np.ndarray
            array of prior predictive distribution evaluated at provided
            test points
        mean : np.ndarray
            average mixed model value at each provided test points
        credible_intervals : np.ndarray
            intervals corresponding for 60%, 90% credible intervals
        std_dev : np.ndarray
            sample standard deviation of mixed model output at provided test
            points
        '''
        if self.prior is None:
            raise Exception("Define the prior first using set_prior")
        else:
            return self.predict(x, CI=[5,95], samples=self.prior.sample(n_sample))

    def set_prior(self, bilby_prior_dic):
        '''
        Set prior for the mixing function parameters.
        Prior for the model parameters should be defined in each model.

        Parametrs:
        ----------
        bilby_prior_dic : bilby.core.prior.PriorDict
            The keys should be named as following :
                '<mix_func_name>_1', '<mix_func_name>_2', ...

        Returns
        -------
        A full Bilby prior object for the mixed model.
        Including the mixing function parameters and model parameters.
        The Bilby prior dictionary has following keys.
            Prior for mixture function parameter :
                '<mix_func_name>_1', '<mix_func_name>_2', ...
            Prior parameters for model 1 : 
                '<name_of_the_model>_1', '<name_of_the_model>_2' , ...
            Prior parameters for model 2 : 
                '<name_of_the_model>_1', '<name_of_the_model>_2' , ...

        '''
        for i, entry in enumerate(self.models_dic):
            model = entry.items()
            name = entry.keys()
            priors = model.prior
            if priors is None:
                continue;
            for ii, entry2 in enumerate(priors):
                bilby_prior_dic.update({f'{name}_{ii}', entry2.items()[ii]})
        self.prior = bilby_prior_dic
        return self.prior

    def mix_loglikelihood(self, mixture_params, model_param, x_exp, y_exp, y_err) -> float:
        """
        log likelihood of the mixed model given the mixing function parameters
        Parameters
        ----------
        mixture_params : np.1darray
            parameter values that fix the shape of mixing function
        model_1_param: np.1darray
            parameter values in the model 1
        model_2_param: np.1darray
            parameter values  in the model 2
        """
        model_1_param , model_2_param = model_param
        W_1, W_2 = self.evaluate_weights(self, mixture_params, x_exp)
        W_1 = np.log(W_1 + eps)
        W_2 = np.log(W_2 + eps)
        L1 = self.models_dic.items()[0].log_likelihood_elementwise(x_exp, y_exp, y_err, model_1_param)
        L2 = self.models_dic.items()[1].log_likelihood_elementwise(x_exp, y_exp, y_err, model_2_param)
        # L1 = log_likelihood_elementwise(self.models_dic.items()[0], self.x_exp, self.y_exp, self.y_err, model_1_param)
        # L2 = log_likelihood_elementwise(self.models_dic.items()[1], self.x_exp, self.y_exp, self.y_err, model_2_param)

        #we use the logaddexp here for numerical accuracy. Look at the 
        #mix_loglikelihood_test to check for an alternative (common) way
        mixed_loglikelihood_elementwise=np.logaddexp(W_1+L1, W_2+L2)
        return np.sum(mixed_loglikelihood_elementwise).item()

    def train(self, x_exp, y_exp, y_err, label='bivariate_mix', outdir='outdir'):
        '''
        Run sampler to learn parameters. Method should also create class
        members that store the posterior and other diagnostic quantities
        important for plotting
        MAP values should also caluclate and set as member variable of
        class

        Return:
        -------
        result : bilby posterior object
            object returned by the bilby sampler
        '''
        priors = self.prior
        if priors is None:
            raise Exception("Please define the priors before training")

        # A few simple setup steps
        likelihood = likelihood_wrapper_for_bilby(self, x_exp, y_exp, y_err)

        if os.path.exists(f'{outdir}/{label}_result.json'):
            os.remove(f'{outdir}/{label}_result.json')
            os.remove(f'{outdir}/{label}_checkpoint_resume.pickle')
            os.remove(f'{outdir}/{label}_samples.txt')
            #shutil.rmtree(outdir)
        result = bilby.run_sampler(
            likelihood,
            priors,
            label=label,
            outdir=outdir,
            sampler="ptemcee",
            ntemps=10,
            nwalkers=20,
            Tmax=100,
            nburn=200,
            nsamples=3000,  # This is the number of raw samples
            threads=5,
            )
        self.posterior = result.posterior.values[:,0:-2]
        self.model_was_trained = True
        return result


