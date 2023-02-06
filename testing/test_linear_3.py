import _mypackage

import numpy as np
import matplotlib.pyplot as plt

from typing import Union

from Taweret.core.base_model import BaseModel
from Taweret.mix.linear import LinearMixerLocal

import my_plotting as mp

# Basic example form Coleman thesis for parameter inference
# Then in a separate file, an example using the SAMBA models


class ModelLine(BaseModel):
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept

    def evaluate(self, x):
        return self.slope * x + self.intercept

    def log_likelihood_elementwise(self, x, y_data, y_err):
        y_theor = self.evaluate(x)
        ll = np.exp(-(y_theor - y_data) ** 2 / (2 * y_err ** 2))
        ll /= np.sqrt(2 * np.pi * y_err ** 2)
        return np.sum(np.log(ll))

    def set_prior(self):
        pass


def coleman_truth(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return 2 - 0.1 * (x - 4) ** 2


def coleman_thesis_example():
    model_1 = ModelLine(slope=0.5, intercept=0.0)
    model_2 = ModelLine(slope=-0.5, intercept=4.0)

    models = {'model 1': model_1, 'model 2': model_2}
    local_linear_mixer = LinearMixerLocal(models=models,
                                          n_mix=1)
    local_linear_mixer.set_prior(
        local_parameter_ranges=np.asarray([-1, 9])
    )

    xs = np.linspace(-1, 9, 10)
    y_truth = coleman_truth(xs)
    y_data = np.random.normal(loc=y_truth, scale=1.0)

    # Currently what the code seems to do is walk throught the space (here
    # space means the space spanned by the local variables) as opposed to
    # adjusting the value of the hyperparameters of the RBF.
    # The hope is that enough steps are taken to get a good sampl
    # acceptance/rejection rate going at that point in local variables
    #
    # FIXME: This means tha the set_prior function needs to be modified to
    #        reflect which parameter is being MCMC walked
    posterior = local_linear_mixer.train(y_exp=y_data,
                                         y_err=1.0,
                                         initial_local_parameters=xs[5],
                                         burn=5,
                                         steps=20,
                                         thinning=1)

    print(posterior.shape)
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(2 * 7, 7))
    fig.patch.set_facecolor('white')

    weights = np.vstack(
        [
            local_linear_mixer.evaluate_weights(local_parameters=x,
                                                number_samples=1)
            for x in posterior.reshape(-1, 1)
        ]
    )
    print(weights)


if __name__ == "__main__":
    coleman_thesis_example()
