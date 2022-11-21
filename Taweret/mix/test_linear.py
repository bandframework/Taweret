import _mypackage

import linear
import numpy as np

from Taweret.core.base_model import BaseModel


class Model(BaseModel):
    def evaluate(self):
        return 1.0

    def log_likelihood_elementwise(self):
        pass

    def set_prior(self):
        pass


if __name__ == "__main__":
    dummy_models = [Model() for i in range(5)]
    global_linear_mix = linear.LinearMixerGlobal(models=dummy_models,
                                                 n_mix=len(dummy_models))
    global_linear_mix.set_prior()
    samples = global_linear_mix.sample_prior(num_samples=10)
    prior_pred = global_linear_mix.prior_predict(num_samples=10,
                                                 model_params=[])

    y_exp = 1.0
    y_err = 0.1
    post = global_linear_mix.train(y_exp=y_exp, y_err=y_err, model_params=[])
    print(post.shape)
