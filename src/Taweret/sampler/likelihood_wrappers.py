# Add any new likelihood wrappers here to be used with other calibration
# packages
import bilby
import numpy as np


class likelihood_wrapper_for_bilby(bilby.Likelihood):


    def __init__(self, mixed_model, x_exp, y_exp, y_err):

        """
        Likelihood wrapper for Bilby calibrator
        ...

        Attributes:
        -----------
        mixed_model : object
            mixed model object from Taweret model mixing class
        x_exp : array
            The independent variables for the experimental data.
        y_exp : array
            The dependent variables for the data.
        y_err : array
            The error bands on the data.
            
        Methods:
        --------
        log_likelihood(self)
            calculates the log likelihood for the parameter values specefied in
            the wrapper object.
        """

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
        The log likelihood function that can be used with Bilby.

        Parameters:
        -----------
        None.

        Returns:
        --------
            The scalar log likelihood value.

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
