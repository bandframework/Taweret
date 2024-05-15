# Add any new likelihood wrappers here to be used with other calibration
# packages
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

    def __init__(self, mixed_model, x_exp, y_exp, y_err):

        param_dic = {i: None
                     for i in mixed_model.prior.keys()
                     }

        super().__init__(parameters=param_dic)
        self.mixed_model = mixed_model
        self.x_exp = x_exp
        self.y_exp = y_exp
        self.y_err = y_err

    def log_likelihood(self):
        """
        log likelihood function that can be used with Bilby.
        return the scalar log likelihood value.
        """
        params = list(self.parameters.values())

        # Because when putting consteaints dummy variables enter which are None
        # in parameters
        params = [i for i in params if i is not None]
        params = np.array(params).flatten()

        mix_param = params[0:self.mixed_model.n_mix]
        models_params = []
        tot_sum = self.mixed_model.n_mix
        n_args_list = list(self.mixed_model.nargs_model_dic.values())
        for i in range(0, len(n_args_list)):
            models_params.append(params[tot_sum: tot_sum + n_args_list[i]])
            tot_sum += n_args_list[i]
            if self.mixed_model.same_parameters:
                break

        return self.mixed_model.mix_loglikelihood(
            mix_param, models_params, self.x_exp, self.y_exp, self.y_err)
