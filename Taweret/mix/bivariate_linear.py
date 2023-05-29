import bilby
import os
import shutil
import numpy as np
from Taweret.utils.utils import mixture_function, eps, normed_mvn_loglike
from Taweret.core.base_mixer import BaseMixer
from Taweret.core.base_model import BaseModel
from Taweret.sampler.likelihood_wrappers import likelihood_wrapper_for_bilby

class BivariateLinear(BaseMixer):

    '''
    Local linear Bayesian mixing of two models. This is a general class of mixing that 
    offer both density (likleihood) and mean mixing methods. The default mixing method
    is linear mixing of two likelihoods (BMMoC). 
    '''

    def __init__(self, models_dic, method='sigmoid', nargs_model_dic=None, 
                 same_parameters = False, cross_corr=False, mean_mix=False):
        '''
        Parameters
        ----------
        models_dic : dictionary {'name1' : model1, 'name2' : model2}
            Two models to mix, each must be derived from the base_model.
        method : str
            Mixing function
        nargs_model_dic : dictionary {'name1' : N_model1, 'name2' : N_model2}
            Only used in calibration. Number of free parameters in each model
        same_parameters : bool
            Only used in calibration. If set, two models are assumed to have same parameters.
        cross_corre : bool
            Implements BMMcor method of Bayesian model mixing. 
        mean_mix : bool
            If set perform mean mixing instead of density mixing.
        '''

        if nargs_model_dic is None:
            nargs_model_dic = {}
        if not isinstance(nargs_model_dic, dict):
            raise AttributeError("nargs_model_dic has to be a dictionary")
        # check if more than two models are trying to be mixed
        if len(models_dic)!=2:
            raise Exception('Bivariate linear mixing requires only two models.\
                            Please look at the other mixing methods in Taweret \
                            for multi modal mixing')

        #check if the models are derived from the base class
        for i, model in enumerate(list(models_dic.values())):
            try:
                isinstance(model, BaseModel) # model is not a class but an object
            except AttributeError:
                print(f'model {list(models_dic.keys())[i]} is not derived from \
                    taweret.core.base_model class')
            else:
                continue
        self.models_dic = models_dic
        # If a new method is added the following needs to be updated
        # It has method and number of free parameters in each method
        method_n_mix_dic = {'step':1, 'addstep':2, 'addstepasym':3, 'sigmoid':2, 'cdf':2, 
                            'switchcos':3, 'calibrate_model_1':0, 'calibrate_model_2':0}

        #check if the mixing function exist
        if method not in method_n_mix_dic:
            raise Exception('Mixing function is not found')
        else:
            self.n_mix=method_n_mix_dic[method]
            print(f'{method} mixing function has {self.n_mix} free parameter(s)')
        #assign default priors
        priors = bilby.core.prior.PriorDict()
        for i in range(0, self.n_mix):
            name = f'{method}_{i}'
            priors[name]=bilby.core.prior.Uniform(0, 1, name=name)
        print(f'Warning : Default prior is set to {priors}')
        print('To change the prior use `set_prior` method')
        self.same_parameters = same_parameters
        self.method = method
        self.nargs_model_dic = nargs_model_dic
        self.model_was_trained=False # Flag to know if the model was trained or not
        self._map=None
        self._posterior=None
        self._prior=self.set_prior(priors) # This combines model priors with mixing method priors
        self.cross_corr = cross_corr
        self.mean_mix = mean_mix
# Attributes
    @property
    def prior(self):
        return self._prior
    @prior.setter
    def prior(self, bilby_prior_dic):
        return self.set_prior(bilby_prior_dic)
    @property
    def posterior(self):
        if self._posterior is None:
            raise Exception('First train the model to access the posterior')
        else:
            return self._posterior
    @property
    def map(self):
        if self._map is None:
            raise Exception('First train the model to access the MAP')
        else:
            return self._map
#End Attributes

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

        w1, w2 = mixture_function(self.method, x, mixture_params, self.prior)
        model_1, model_2 = list(self.models_dic.values())
        try:
            if self.same_parameters:
                model_1_out,_ = model_1.evaluate(x,model_params[0])
            else:
                model_1_out,_ = model_1.evaluate(x,model_params[0])
        except:
            model_1_out,_ = model_1.evaluate(x)

        try:
            if self.same_parameters:
                model_2_out,_ = model_2.evaluate(x,model_params[0])
            else:
                model_2_out,_ = model_2.evaluate(x,model_params[1])
        except:
            model_2_out,_ = model_2.evaluate(x)        

        #try:
        #    model_1_out,_ = model_1.evaluate(x,model_params[0])
        #except:
        #    model_1_out,_ = model_1.evaluate(x)
        #
        #try:
        #    if self.same_parameters:
        #        model_2_out,_ = model_2.evaluate(x,model_params[0])
        #    else:
        #        model_2_out,_ = model_2.evaluate(x,model_params[1])
        #except:
        #    model_2_out,_ = model_2.evaluate(x)
        
        if model_1_out.ndim == model_2_out.ndim and model_2_out.ndim <= 1:
            return w1*model_1_out + w2*model_2_out
        elif model_1_out.ndim == model_2_out.ndim and model_2_out.ndim == 2:
            if len(model_1_out)!=len(model_2_out):
                raise Exception('Dimension mistmatch between outputs of models')
            outputs = []
            for obs_n in range(0,model_1_out.shape[1]):
                outputs.append(w1*model_1_out[:,obs_n] + w2*model_2_out[:,obs_n]) 
            return np.array(outputs)
        else: 
            assert Exception(f'Dimensional mistmatch: dim of model 1 is {model_1_out.ndim}\
                , model 2 is {model_2_out.ndim}')
    
    
    def evaluate_weights(self, mixture_params, x):
        '''
        return the mixing function values at the input parameter values x

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

        return mixture_function(self.method, x, mixture_params, self.prior)


    def predict(self, x, CI=[5,95], samples=None, nthin=1):
        '''
        Evaluate posterior to make prediction at test points x.

        Parameters
        ----------
        x : np.1darray
            input parameter values
        CI : list
            confidence intervals as percentages
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

        if self.model_was_trained is False and samples is None:
            raise Exception('Posterior is not available to make predictions\n\
                            train the model before predicting')
        pos_predictions = []
        
        if samples is not None:
            print("using provided samples instead of posterior")
            posterior = samples
        else:
            posterior = self._posterior
        n_samples = posterior.shape[0]
        for sample in posterior[::nthin]:
            sample = np.array(sample).flatten()

            mixture_param = sample[0:self.n_mix]
            model_params = []
            n_args_for_models = list(self.nargs_model_dic.values())
            n_args_sum = 0
            for i in range(0,len(n_args_for_models)):
                model_params.append(sample[self.n_mix+n_args_sum:self.n_mix+n_args_sum+n_args_for_models[i]])
                n_args_sum+=n_args_for_models[i]
                if self.same_parameters:
                    break;
                
            value = self.evaluate(mixture_param,x, model_params)
            pos_predictions.append(value)
        pos_predictions = np.array(pos_predictions)

        CIs = np.percentile(pos_predictions,CI, axis=0, keepdims=True)

        mean = np.mean(pos_predictions, axis=0, keepdims=True)

        std_dev = np.std(pos_predictions, axis=0, keepdims=True)

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
            posterior = samples
        else:
            posterior = self._posterior
        for sample in posterior:
            sample = np.array(sample).flatten()

            mixture_param = sample[0:self.n_mix]

            value,_ = self.evaluate_weights(mixture_param,x)
            pos_predictions.append(value)
        pos_predictions = np.array(pos_predictions)
        print(pos_predictions.shape)


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
        if self._prior is None:
            raise Exception("Define the prior first using set_prior")
        else:
            samples = np.array(list(self._prior.sample(n_sample).values())).T
            print(samples.shape)
            return self.predict(x, CI=CI, samples=samples)

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
        for name, model in self.models_dic.items():
            if model.prior is None:
                #print(f'model has no prior {model}')
                continue
            else:
                priors = model.prior
            for ii, entry2 in enumerate(priors):
                bilby_prior_dic[f'{name}_{ii}']=priors[entry2]
            if self.same_parameters:
                break
        self._prior = bilby_prior_dic
        return self._prior

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
        if len(model_param)==0:
            model_1_param = np.array([])
            model_2_param = np.array([])
        else:
            if self.same_parameters:
                model_1_param = model_param[0]
                model_2_param = model_param[0]
            else:
                model_1_param , model_2_param = model_param
        
        W_1, W_2 = self.evaluate_weights(mixture_params, x_exp)
        model_1, model_2 = list(self.models_dic.values())
        if self.cross_corr and not self.mean_mix:
            L1 = model_1.log_likelihood(x_exp, y_exp, y_err, W_1, model_1_param)
            L2 = model_2.log_likelihood(x_exp, y_exp, y_err, W_2, model_2_param)

            return L1 + L2
        
        if self.mean_mix:
            if self.cross_corr is False:
                raise Exception('mean mixing only works with full covariance matrix')
            predictions_1, model_errs_1, cov_mat_1 = model_1.evaluate(x_exp, model_1_param, full_corr=True)
            predictions_2, model_errs_2, cov_mat_2 = model_2.evaluate(x_exp, model_2_param, full_corr=True)
            
            weights = []
            for w in W_1:
                weights.append(w*np.ones(y_exp.shape[1]))
            weights = np.array(weights).flatten()
            predictions_1 = np.array(predictions_1).flatten()
            predictions_2 = np.array(predictions_2).flatten()
            y_exp_all = np.array(y_exp).flatten()
            y_err_all = np.array(y_err).flatten()
            diff = predictions_1 * weights + predictions_2 * (1-weights)
            N = len(y_exp_all)
            #A better optimization yield 5 times the speed.
            #For testing of the method loo at tests.ipynb in notebooks folder.
#             final_cov = np.zeros((N,N))
#             for i in range(0,N):
#                 for j in range(0,N):
#                     final_cov[i,j] = weights[i]*weights[j]*cov_mat_1[i,j] + (1-weights[i])*(1-weights[j])*cov_mat_2[i,j]
            #Comment out the below code. BMM mean mixing does not touch the covariance stucture....yet...
            #w1_mat = np.outer(weights, weights)
            #w2_mat = np.outer(1-weights,1-weights)
            #final_cov = w1_mat * cov_mat_1 + w2_mat * cov_mat_2
            final_cov = cov_mat_1 + cov_mat_2
            #print(np.all(np.isclose(final_cov,final_cov_2)))
            final_cov += np.diag(np.square(y_err_all))
            return normed_mvn_loglike(diff,final_cov)

        else:
            W_1 = np.log(W_1 + eps)
            W_2 = np.log(W_2 + eps)
            if self.method=='calibrate_model_2':
                L1 = np.zeros(len(x_exp))
                L2 = model_2.log_likelihood_elementwise(x_exp, y_exp, y_err, model_2_param)
            elif self.method=='calibrate_model_1':
                L2 = np.zeros(len(x_exp))
                L1 = model_1.log_likelihood_elementwise(x_exp, y_exp, y_err, model_1_param)
            else:
                L1 = model_1.log_likelihood_elementwise(x_exp, y_exp, y_err, model_1_param)
                L2 = model_2.log_likelihood_elementwise(x_exp, y_exp, y_err, model_2_param)
            # L1 = log_likelihood_elementwise(self.models_dic.items()[0], self.x_exp, self.y_exp, \
            # self.y_err, model_1_param)
            # L2 = log_likelihood_elementwise(self.models_dic.items()[1], self.x_exp, self.y_exp, \
            # self.y_err, model_2_param)

            #we use the logaddexp here for numerical accuracy. Look at the
            #mix_loglikelihood_test to check for an alternative (common) way
            mixed_loglikelihood_elementwise=np.logaddexp(W_1+L1, W_2+L2)
            return np.sum(mixed_loglikelihood_elementwise).item()


    def train(self, x_exp, y_exp, y_err, label='bivariate_mix', outdir='outdir', 
    kwargs_for_sampler=None, load_previous=False):
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
        prior = self._prior
        if prior is None:
            raise Exception("Please define the priors before training")

        # A few simple setup steps
        likelihood = likelihood_wrapper_for_bilby(self, x_exp, y_exp, y_err)

        #if os.path.exists(outdir) and load_previous:
        try:
            
            result = bilby.result.read_in_result(outdir=outdir, label=label)
        except:
            if load_previous:
                print(f'Saved results for {label} do not exist in : '+ outdir)
            #if os.path.exists(outdir+'/'+label):
            #    shutil.rmtree(outdir+'/'+label)
            if kwargs_for_sampler is None:
                kwargs_for_sampler = {'sampler':'ptemcee',
                'ntemps':10,
                'nwalkers':200,
                'Tmax':100,
                'burn_in_fixed_discard':5000,
                'nsamples':20000,
                'threads':28,
                'printdt':60}
                #'safety':2,
                #'autocorr_tol':5}
                print(f'The following Default settings for sampler will be used. You can change\
    these arguments by providing kwargs_for_sampler argement in `train`.\
    Check Bilby documentation for other sampling options.\n{kwargs_for_sampler}')
            else:
                print(f'The following settings were provided for sampler \n{kwargs_for_sampler}')

            result = bilby.run_sampler(
                likelihood,
                prior,
                label=label,
                outdir=outdir,
                **kwargs_for_sampler)
        # The last two columns are model liklihood and log_prior.
        self._posterior = result.posterior.values[:,0:-2]  
        self.model_was_trained = True
        # Shorcut to find MAP. Need to implement a proper optimization routine to find MAP
        self._map=self._posterior[np.argmax(result.posterior.values[:,-2].flatten()),:]

        return result
