import numpy as np
from scipy.stats import norm

__all__ = ["Priors"]


class Priors:

    def __init__(self):
        '''
        Prior class for the mixing functions.
        '''
        return None

    def luniform(self, theta, a, b):
        '''
        General uniform prior to be used to truncate the normal
        distributions used in the parameter priors, taken from
        https://www.github.com/asemposki/SAMBA/samba/priors.py.
        '''

        if theta > a and theta < b:
            return 0.0
        else:
            return -np.inf

    def lpdf(self, params):
        '''
        Log pdf of the priors for the parameters for the SAMBA model,
        taken directly from https://www.github.com/asemposki/SAMBA/
        samba/priors.py.
        Must be truncated for the sampler to walk in valid regions.
        '''

        if isinstance(params, float):
            params = np.array([params])

        if len(params) == 1:
            param_1 = self.luniform(params[0], 0.0, 1.0)

            return param_1

        if len(params) == 2:
            param_1 = norm.logpdf(params[0], 10.0, 2.0)
            param_2 = norm.logpdf(params[1], -20.0, 10.0)

            return (param_1 + param_2)

        elif len(params) == 3:

            g1 = self.luniform(params[0], 0.01, 0.3) + \
                norm.logpdf(params[0], 0.1, 0.05)

            g3 = self.luniform(params[2], params[0], 0.55) + \
                norm.logpdf(params[2], 0.4, 0.05)

            g2 = self.luniform(params[1], params[2], 0.8) + \
                norm.logpdf(params[1], 0.6, 0.05)

            return (g1 + g2 + g3)

        else:
            raise ValueError('The number of parameters \
            does not match any available switching function.')
