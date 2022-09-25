# Add any new likelihood wrappers here to be used with other calibration packages
import bilby
import numpy as np

class likelihood_wrapper_for_bilby(bilby.Likelihood):
    """
    Likelihood wrapper for Bilby calibrator
    ...

    Attributes
    ----------
    mixed_model : object
        mixed model object from Taweret model mixing class
    parameters : dic
        dictionary with name of each parameter to be given to Bilby.
        all parameters have the prefix theta_ and then the parameter number.
        ex : for liklihood with three free parameters ; theta_0, theta_1, theta_2
        Parameters are order as [mixture function parameters, model 1 parameters, model 2 parameters]

    Methods
    -------
    log_likelihood(self)
        calculates the log likelihood for the parameter values specefied in the wrapper object. 
    """


    def __init__(self, mixed_model):
        
        param_dic={}
        for i in range(0, mixed_model.n_model_1 + mixed_model.n_model_2 + mixed_model.n_mix):
            param_dic[f'theta_{i}']=None

        super().__init__(parameters=param_dic)
        self.mixed_model=mixed_model

    def log_likelihood(self):
        """
        log likelihood function that can be used with Bilby.
        return the scalar log likelihood value.
        """

        params = list(self.parameters.values())
        params = np.array(params).flatten()

        mix_param = params[0:self.mixed_model.n_mix]
        m1_param = params[self.mixed_model.n_mix:self.mixed_model.n_model_1]
        m2_param = params[-self.mixed_model.n_model_2:]

        return self.mixed_model.mix_loglikelihood(mix_param, m1_param, m2_param)