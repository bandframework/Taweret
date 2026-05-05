# Written by: Alexandra C. Semposki, Jordan A. Melendez
# Original code from the neutron-rich-bmm repository

# necessary imports
from Taweret.core.base_mixer import BaseMixer
import numpy as np
from scipy import stats
import sys

# imports for the sklearn interface
from operator import itemgetter
from scipy.linalg import cho_solve, cholesky
import scipy as scipy
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.base import clone
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y

sys.path.append("../Taweret")

# global cholesky setting
GPR_CHOLESKY_LOWER = True


# set up classes for each option here
class GPmixing(BaseMixer):

    def __init__(
        self, x, models, alpha=None, mean_function="zero", kernel=None,
        priors=True, prior_params=None, prior_choice=None, prior_type=None,
        switch=None, max_iter=None, nopt=1000
    ):
        """
        Parameters:
        -----------
        x : numpy.linspace
            Input space variable in which mixing is occurring.

        models : dict
            Dict of models with BaseModel methods.

        mean_function: str
            Selection of the mean function chosen for the GP.
            Choices include: 'zero' and 'spline'.

        kernel: str
            Choice of the kernel to be used in the GP. Choices
            include stationary and non-stationary kernels: 'rbf',
            'matern32', 'matern52', and 'rq' for stationary;
            'sigmoid', 'tanh', or 'theta' for non-stationary.

        priors: dict
            Dict of hyperpriors for the selected kernel. Default
            priors included in the package will be run if there
            are no specified hyperpriors, depending on the kernel
            selected.

        Returns:
        --------
        None.
        """

        # set up the class variables assuming models are valid
        self.model_dict = models
        self.x = x
        self.alpha = alpha
        self.nopt = nopt

        # initialize the trained state to False
        self._is_trained = False

        # max iterations for the optimizer
        self.max_iter = max_iter

        # str class variables
        self.mean_function_choice = mean_function
        self.priors = priors  # if False, will use LML instead of MAP
        self.prior_params = prior_params
        self.prior_choice = prior_choice
        self.prior_type = prior_type  # only used for changepoint kernel
        self.switch = switch

        # handle None case for kernel
        if kernel is None:
            self.kernel = C(1.0, constant_value_bounds="fixed") * RBF(
                1.0, length_scale_bounds="fixed"
            )
        else:
            self.kernel = clone(kernel)

        # make a copy of the unconstrained kernel for use later
        self.kernel_copy = self.kernel

        # get hyperpriors set up here
        if self.prior_params is None and self.priors is True:
            self.prior_choice = 'rbfnorm'
            self.prior_params = self._default_prior_params()

        # convert models dict() to list
        self.models = [i for i in self.model_dict.values()]

        return None

    def evaluate(self, x):
        """
        Evaluation of the model at the set of training points. If
        the model was not yet fit, this function will return the
        result of the prior at the training points.
        """

        # set up the training point array
        if x.ndim == 1:
            x_eval = x.reshape(-1, 1)
        else:
            x_eval = x

        # evaluate at a selected array of points
        if self._is_trained is True:
            eval_mean, eval_std = self.gpr.predict(x_eval, return_std=True)
            _, eval_cov = self.gpr.predict(x_eval, return_cov=True)
        else:
            # call wrapper with unconstrained kernel
            unconstrained_prior = GPRwrapper(
                kernel=self.kernel_copy,
                alpha=self.alpha,
                n_restarts_optimizer=self.nopt
            )

            # now calculate results of the GP prior (no fitting!)
            eval_mean, eval_std = unconstrained_prior.predict(x_eval,
                                                              return_std=True)
            _, eval_cov = unconstrained_prior.predict(x_eval, return_cov=True)

        # collect results
        eval_results = {
                "x": x_eval[:, 0],
                "mean": eval_mean,
                "std": eval_std,
                "cov": eval_cov,
            }

        return eval_results

    def evaluate_weights(self):
        """
        Evaluation of a point estimate of the weights
        of the models used. Not able to be done for this
        mixing method since GP is implicitly weighted.
        """
        return NotImplemented

    @property
    def map(self):
        """
        Return the MAP values of the parameters,
        exponentiated for readability as the true values
        of the parameters.
        """

        if self._is_trained is True:
            return np.exp(self.gpr.kernel_.theta)
        else:
            raise ValueError('You need to train the GP first.')

    @property
    def posterior(self):
        """
        Return the posterior of the parameters.
        Not needed for this mixing method; we
        only return the MAP currently.
        """
        return None

    def predict(self):
        """
        Here the GP needed to perform the mixing is predicted
        at the requested points.

        Parameters:
        -----------
        ci : int, list
            The desired credibility interval(s) (1-sigma, 2-sigma)

        Returns:
        --------
        gp_results : dict
            A dict of the prediction points in the input space,
            and the means, variances, and covariances at each of
            these specified locations.
        """

        if self._is_trained is True:
            self.mean, self.std_dev = self.gpr.predict(self.x.reshape(-1, 1),
                                                       return_std=True)
            _, self.cov = self.gpr.predict(self.x.reshape(-1, 1),
                                           return_cov=True)
        else:
            raise ValueError("You must train the model first.")

        # set up the dict of values to return
        gp_results = {
            "x": self.x,
            "mean": self.mean,
            "std": self.std_dev,
            "cov": self.cov,
        }

        return gp_results

    def predict_weights(self):
        """
        Predict the weights of the mixed model. Returns
        mean and intervals from the posterior of the
        weights. Cannot predict weights for GP implicitly.
        """
        return NotImplemented

    @property
    def prior(self):
        """
        Return the prior of the parameters in the mixing.
        Not needed for this method, since hyperpriors are
        really what we are using.
        """
        return None

    def prior_predict(self, sample=False, n_samples=None):
        """
        Find the predicted prior distribution using the
        unconstrained GP (e.g., no hyperparameter optimization
        has been performed yet).
        """

        # call wrapper with unconstrained kernel
        unconstrained_prior = GPRwrapper(
            kernel=self.kernel_copy,
            alpha=self.alpha,
            n_restarts_optimizer=self.nopt
        )

        # now calculate results of the GP prior (no fitting!)
        x_pred = self.x.reshape(-1, 1)
        prior_mean, prior_std = unconstrained_prior.predict(x_pred,
                                                            return_std=True)
        _, prior_cov = unconstrained_prior.predict(x_pred, return_cov=True)

        prior_results = {
            "x": self.x,
            "mean": prior_mean,
            "std": prior_std,
            "cov": prior_cov,
        }

        # if sampling return samples; if not, return dict
        if sample is True:
            samples = unconstrained_prior.sample_y(x_pred, n_samples=n_samples)
            return prior_results, samples
        else:
            return prior_results

    def sample_prior(self):
        """
        Returns samples from the prior
        distributions for the various weight parameters.
        Not needed for this mixing method.
        """
        return NotImplemented

    def set_prior(self):
        """
        Set the priors on the parameters. This will
        be needed for this method but not yet determined how
        to properly use it.
        """
        return None

    def train(self, X, y, prior_choice='rbfnorm', prior_type=None,
              switch=None, max_iter=None):
        """
        Train the GP chosen in the __init__() function
        to optimize its hyperparameters given chosen priors
        and models. Needs to be implemented.

        Parameters:
        -----------
        X (array-like of shape (n_samples, n_features) or list of object):
            Feature vectors or other representations of training data.

        y (array-like of shape (n_samples,) or (n_samples, n_targets)):
            Target values.

        prior_choice (str): The choice of which type of prior to use on
            the length scale. Default is 'rbfnorm'; other options are
            'skewnorm' and 'uniform'.

        prior_type (dict): The type of prior we want to use on the
            hyperparameters when in a situation where more than one
            hyperparameter will be optimized, or when we do not want
            to use a log normal prior on the chosen hyperparameter
            in the changepoint kernel. This also takes in the switching
            function type; currently options are 'tanh' and 'sigmoid'.

        switch (str): If using a changepoint kernel, specify which
            switching function you are using. Default is None
            to indicate not using this kernel.
        """

        # fit function call from GPRWrapper
        self.gpr = GPRwrapper(kernel=self.kernel, alpha=self.alpha,
                              n_restarts_optimizer=self.nopt)
        self.gpr_obj = self.gpr.fit(X, y,
                                    priors=self.priors,
                                    prior_choice=prior_choice,
                                    prior_params=self.prior_params,
                                    prior_type=prior_type, switch=switch,
                                    max_iter=max_iter)

        # make sure it is clear it has been trained
        self._is_trained = True

        return None

    def _default_prior_params(self):
        """
        Default prior in case the user has no idea what
        to use and would like to play with a pre-built case.
        ** Params array is in log space because
        the GP code requires this and cannot be changed. **
        """

        # check if guesses are provided for bounds; if not, use local
        if self.prior_params is None and self.priors is True:
            if self.prior_choice == 'rbfnorm':
                self.prior_params = {
                    "sigma": {"mu": 1.0, "sig": 0.25},
                    "lengthscale": {"mu": 1.0, "sig": 0.15}
                }

        return self.prior_params


class GPRwrapper(GaussianProcessRegressor):

    def fit(self, X, y, priors=True, prior_choice='truncnorm',
            prior_type=None, prior_params=None, switch=None, max_iter=None):
        """
        Meat of the GP training, where we use the GPR
        class in scikit-learn and wrap it with this
        treatment of the extracted model means, variances,
        and covariances.
        """

        # set up the class variables here
        self.priors = priors
        self.prior_choice = prior_choice
        self.prior_type = prior_type
        self.prior_params = prior_params
        self.switch = switch
        self.max_iter = max_iter

        # kernel should not be None but still
        if self.kernel is None:
            self.kernel_ = C(1.0, constant_value_bounds="fixed") * RBF(
                1.0, length_scale_bounds="fixed"
            )
        else:
            self.kernel_ = clone(self.kernel)

        # we need to determine why these variables are not passing
        self._rng = check_random_state(self.random_state)

        if self.kernel_.requires_vector_input:
            dtype, ensure_2d = "numeric", True
        else:
            dtype, ensure_2d = None, False
        X, y = check_X_y(
            X,
            y,
            multi_output=True,
            y_numeric=True,
            ensure_2d=ensure_2d,
            dtype=dtype,
        )

        # added because it is not passing this for some reason from sklearn
        n_targets = None
        self.n_targets = n_targets

        n_targets_seen = y.shape[1] if y.ndim > 1 else 1
        if self.n_targets is not None and n_targets_seen != self.n_targets:
            raise ValueError(
                """The number of targets seen in `y` is different from the
                initial"""
                f"`n_targets`. Got {n_targets_seen} != {self.n_targets}."
            )

        # shape correctly; no normalizing here b/c of covariance structure
        shape_y_stats = (y.shape[1],) if y.ndim == 2 else 1
        self._y_train_mean = np.zeros(shape=shape_y_stats)
        self._y_train_std = np.ones(shape=shape_y_stats)

        # alpha is the covariance of the model
        if np.iterable(self.alpha) and self.alpha.shape[0] != y.shape[0]:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError(
                    "alpha must be a scalar or an array with same number of "
                    f"entries as y. ({self.alpha.shape[0]} != {y.shape[0]})"
                )

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

        # set the hyperpriors here
        if self.priors is True:
            self.prior_ = GPPriors(
                kernel=self.kernel_,
                prior_choice=self.prior_choice,
                prior_type=self.prior_type,
                prior_params=self.prior_params,
                switch=self.switch,
            )

        # LML or MAP, depending on state of the prior
        if self.optimizer is not None and self.kernel_.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True):

                # MAP value case with priors included
                if self.priors is True:
                    if eval_gradient:
                        lml, grad_lml = self._log_marginal_likelihood(
                            theta, eval_gradient=True, clone_kernel=False
                        )
                        lp, grad_lp = self.prior_.log_priors(theta)
                        return -(lml + lp), -(grad_lml + grad_lp)

                    else:
                        lml = self._log_marginal_likelihood(theta,
                                                            clone_kernel=False)
                        lp, _ = self.prior_.log_priors(theta)
                        return -(lml + lp)

                # handle LML case only if no priors selected
                else:
                    if eval_gradient:
                        lml, grad_lml = self._log_marginal_likelihood(
                            theta, eval_gradient=True, clone_kernel=False
                        )
                        return -lml, -grad_lml

                    else:
                        lml = self._log_marginal_likelihood(theta,
                                                            clone_kernel=False)
                        return -lml

            # First optimize starting from theta specified in kernel
            optima = [
                self._constrained_optimization(
                    obj_func,
                    self.kernel_.theta,
                    self.kernel_.bounds,
                    max_iter=self.max_iter,
                )
            ]

            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite."
                    )
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = self._rng.uniform(bounds[:, 0],
                                                      bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(
                            obj_func, theta_initial, bounds,
                            max_iter=self.max_iter
                        )
                    )
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.kernel_._check_bounds_params()

            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = (
                self._log_marginal_likelihood(
                    self.kernel_.theta,
                    clone_kernel=False,
                )
            )

        # Precompute quantities required for predictions which are independent
        # of actual query points
        # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
        K = self.kernel_(self.X_train_)
        # Handle 2d noise:
        if np.iterable(self.alpha) and self.alpha.ndim == 2:
            K += self.alpha
        else:
            K[np.diag_indices_from(K)] += self.alpha

        try:
            self.L_ = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError as exc:
            exc.args = (
                (
                    f"The kernel, {self.kernel_}, is not returning a positive "
                    "definite matrix. Try gradually increasing the 'alpha' "
                    "parameter of your GaussianProcessRegressor estimator."
                ),
            ) + exc.args
            raise
        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        self.alpha_ = cho_solve(
            (self.L_, GPR_CHOLESKY_LOWER),
            self.y_train_,
            check_finite=False,
        )

        return self

    def _log_marginal_likelihood(
        self, theta=None, eval_gradient=False, clone_kernel=True
    ):
        """Return log-marginal likelihood of theta for training data.

        Parameters:
            theta (array-like of shape (n_kernel_params,) default=None):
                Kernel hyperparameters for which the log-marginal likelihood is
                evaluated. If None, the precomputed log_marginal_likelihood
                of ``self.kernel_.theta`` is returned.

            eval_gradient (bool, default=False): If True, the gradient of the
                log-marginal likelihood with respect
                to the kernel hyperparameters at position theta is returned
                additionally. If True, theta must not be None.

            clone_kernel (bool, default=True): If True, the kernel attribute
                is copied. If False, the kernel
                attribute is modified, but may result in a performance
                improvement.

        Returns:
            log_likelihood (float): Log-marginal likelihood of theta for
                training data.

            log_likelihood_gradient (ndarray of shape (n_kernel_params,),
                optional): Gradient of the log-marginal likelihood with
                respect to the kernel hyperparameters at position theta.
                Only returned when eval_gradient is True.
        """
        if theta is None:
            if eval_gradient:
                raise ValueError("""Gradient can only be evaluated
                                for theta!=None""")
            return self.log_marginal_likelihood_value_

        if clone_kernel:
            kernel = self.kernel_.clone_with_theta(theta)
        else:
            kernel = self.kernel_
            kernel.theta = theta

        if eval_gradient:
            K, K_gradient = kernel(
                self.X_train_, eval_gradient=True
            )  # already evaluated here!
            self.K_copy = K
            self.K_copy_gradient = K_gradient
        else:
            K = kernel(self.X_train_, eval_gradient=False)
            self.K_copy = K

        # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
        # Handle 2d noise:
        if np.iterable(self.alpha) and self.alpha.ndim == 2:
            K += self.alpha
            # print('K with alpha, K shape: ', K, K.shape)
            self.Kalph = K
        else:
            K[np.diag_indices_from(K)] += self.alpha
        try:
            L = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError:
            if eval_gradient:
                return (-np.inf, np.zeros_like(theta))
            else:
                return -np.inf

        # Support multi-dimensional output of self.y_train_
        y_train = self.y_train_
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]

        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        alpha = cho_solve((L, GPR_CHOLESKY_LOWER), y_train, check_finite=False)

        # Alg 2.1, page 19, line 7
        # -0.5 . y^T . alpha - sum(log(diag(L))) - n_samples / 2 log(2*pi)
        # y is originally thought to be a (1, n_samples) row vector. However,
        # in multioutputs, y is of shape (n_samples, 2) and we need to compute
        # y^T . alpha for each output, independently using einsum. Thus, it
        # is equivalent to:
        # for output_idx in range(n_outputs):
        #     log_likelihood_dims[output_idx] = (
        #         y_train[:, [output_idx]] @ alpha[:, [output_idx]]
        #     )
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
        self.piece_1 = log_likelihood_dims
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        self.piece_2 = -np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        self.piece_3 = -K.shape[0] / 2 * np.log(2 * np.pi)
        # the log likehood is sum-up across the outputs     ---- > here we
        # should be able to dissect each piece
        log_likelihood = log_likelihood_dims.sum(axis=-1)

        if eval_gradient:
            # Eq. 5.9, p. 114, and footnote 5 in p. 114
            # 0.5 * trace((alpha . alpha^T - K^-1) . K_gradient)
            # alpha is supposed to be a vector of (n_samples,) elements. With
            # multioutputs, alpha is a matrix of size (n_samples, n_outputs).
            # Therefore, we want to construct a matrix of
            # (n_samples, n_samples, n_outputs) equivalent to
            # for output_idx in range(n_outputs):
            #     output_alpha = alpha[:, [output_idx]]
            #     inner_term[..., output_idx] = output_alpha @ output_alpha.T
            inner_term = np.einsum("ik,jk->ijk", alpha, alpha)
            # compute K^-1 of shape (n_samples, n_samples)
            K_inv = cho_solve(
                (L, GPR_CHOLESKY_LOWER), np.eye(K.shape[0]), check_finite=False
            )
            # create a new axis to use broadcasting between inner_term and
            # K_inv
            inner_term -= K_inv[..., np.newaxis]
            # Since we are interested about the trace of
            # inner_term @ K_gradient, we don't explicitly compute the
            # matrix-by-matrix operation and instead use an einsum. Therefore
            # it is equivalent to:
            # for param_idx in range(n_kernel_params):
            #     for output_idx in range(n_output):
            #         log_likehood_gradient_dims[param_idx, output_idx] = (
            #             inner_term[..., output_idx] @
            #             K_gradient[..., param_idx]
            #         )
            log_likelihood_gradient_dims = 0.5 * np.einsum(
                "ijl,jik->kl", inner_term, K_gradient
            )
            # the log likelihood gradient is the sum-up across the outputs
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(axis=-1)

        # create class variable
        self.log_likelihood = log_likelihood

        # return results
        if eval_gradient is True:
            return log_likelihood, log_likelihood_gradient
        else:
            return log_likelihood

    def _constrained_optimization(
        self, obj_func, initial_theta, bounds, max_iter
    ):  # added max_iter
        if self.optimizer == "fmin_l_bfgs_b":
            if max_iter is None:
                opt_res = scipy.optimize.minimize(
                    obj_func,
                    initial_theta,
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds,
                    tol=1e-12,  # 1e-12
                )
            else:
                opt_res = scipy.optimize.minimize(
                    obj_func,
                    initial_theta,
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds,
                    tol=1e-12,  # 1e-12
                    options={"maxiter": max_iter},  # added options
                )
            self._check_optimize_result(
                "lbfgs", opt_res, max_iter=max_iter
            )  # added max_iter
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta,
                                                 bounds=bounds)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}.")

        return theta_opt, func_min

    def _check_optimize_result(
        self, solver, result, max_iter=None, extra_warning_msg=None
    ):
        """Check the OptimizeResult for successful convergence

        Parameters:
            solver (str): Solver name. Currently only `lbfgs` is supported.

            result (OptimizeResult): Result of the scipy.optimize.minimize
                function.

            max_iter (int, default=None): Expected maximum number of
                iterations.

            extra_warning_msg (str, default=None): Extra warning message.

        Returns:
            n_iter (int): Number of iterations.
        """
        # handle both scipy and scikit-learn solver names
        if solver == "lbfgs":
            if result.status != 0:
                try:
                    # The message is already decoded in scipy>=1.6.0
                    result_message = result.message.decode("latin1")
                except AttributeError:
                    result_message = result.message
                warning_msg = (
                    "{} failed to converge (status={}):\n{}.\n\n"
                    "Increase the number of iterations (max_iter) "
                    "or scale the data as shown in:\n"
                    "    https://scikit-learn.org/stable/modules/"
                    "preprocessing.html"
                ).format(solver, result.status, result_message)
                if extra_warning_msg is not None:
                    warning_msg += "\n" + extra_warning_msg
                warnings.warn(
                    warning_msg
                )  # ConvergenceWarning, stacklevel=2)
            if max_iter is not None:
                # In scipy <= 1.0.0, nit may exceed maxiter for lbfgs.
                # See https://github.com/scipy/scipy/issues/7854
                n_iter_i = min(result.nit, max_iter)
            else:
                n_iter_i = result.nit
        else:
            raise NotImplementedError

        return n_iter_i


class GPPriors:

    # allow for stationary and non-stationary options
    def __init__(self, kernel, prior_choice=None, prior_type=None,
                 prior_params=None, switch=None):
        self.kernel_ = kernel
        self.prior_choice = prior_choice
        self.prior_type = prior_type
        self.prior_params = prior_params
        self.switch = switch

        return None

    # chunky function that could be improved and split later
    def log_priors(self, theta, **kwargs):

        if self.prior_choice == 'changepoint':

            # will look for both of these as a default
            cp_bounds = True
            w_bounds = True

            # make dict of values for this part to tell code which is optimized
            if self.kernel_.width_bounds == 'fixed':
                w_bounds = False
            if self.kernel_.changepoint_bounds == 'fixed':
                cp_bounds = False
            arg_dict = {
                'cp_opt': cp_bounds,
                'w_opt': w_bounds
            }

            # include both changepoint and width options
            cp_opt = arg_dict['cp_opt']
            w_opt = arg_dict['w_opt']

            # means and variances of width
            if self.switch == 'sigmoid':
                mean_w = 0.16
                var_w = 0.155
            elif self.switch == 'tanh':
                mean_w = 0.32
                var_w = 0.31

            # optimizing both parameters
            if cp_opt is True and w_opt is True:

                # converting back into parameter space
                cp = np.exp(theta[0])
                w = np.exp(theta[1])

                cpa = np.exp(self.kernel_.bounds[0, 0])
                cpb = np.exp(self.kernel_.bounds[0, 1])
                wa = np.exp(self.kernel_.bounds[1, 0])
                wb = np.exp(self.kernel_.bounds[1, 1])

                # construct the prior and gradient
                if (self.prior_type['cp'] == 'truncnorm'
                        and self.prior_type['w'] == 'truncnorm'):
                    deriv_cp_norm = self.deriv_cp(cp)
                    if self.switch == 'sigmoid':
                        deriv_w_norm = self.deriv_w_sigmoid(w)
                    elif self.switch == 'tanh':
                        deriv_w_norm = self.deriv_w_tanh(w)
                    log_prior = self.luniform_ls(cp, cpa, cpb) + \
                        stats.norm.logpdf(cp, 0.98, 0.33) + \
                        self.luniform_ls(w, wa, wb) + \
                        stats.norm.logpdf(w, mean_w, var_w)
                    log_gradient = deriv_cp_norm + deriv_w_norm

                elif (self.prior_type['cp'] == 'truncnorm'
                        and self.prior_type['w'] == 'free'):
                    deriv_cp_norm = self.deriv_cp(cp)
                    log_prior = self.luniform_ls(cp, cpa, cpb) + \
                        stats.norm.logpdf(cp, 0.98, 0.33) + \
                        self.luniform_ls(w, wa, wb)
                    log_gradient = deriv_cp_norm

                elif (self.prior_type['w'] == 'truncnorm'
                      and self.prior_type['cp'] == 'free'):
                    if self.switch == 'sigmoid':
                        deriv_w_norm = self.deriv_w_sigmoid(w)
                    elif self.switch == 'tanh':
                        deriv_w_norm = self.deriv_w_tanh(w)
                    log_prior = self.luniform_ls(cp, cpa, cpb) + \
                        self.luniform_ls(w, wa, wb) + \
                        stats.norm.logpdf(w, mean_w, var_w)
                    log_gradient = deriv_w_norm

                elif (self.prior_type['cp'] == 'free'
                      and self.prior_type['w'] == 'free'):
                    log_prior = (self.luniform_ls(cp, cpa, cpb)
                                 + self.luniform_ls(w, wa, wb))
                    log_gradient = 0.0

                # save prior values
                self.log_prior_vals = log_prior
                self.log_prior_grad = log_gradient

                return log_prior, log_gradient

            # optimizing width only
            elif w_opt is True and cp_opt is not True:

                w = np.exp(theta[0])
                wa = np.exp(self.kernel_.bounds[0, 0])
                wb = np.exp(self.kernel_.bounds[0, 1])

                # construct the prior and gradient
                if self.prior_type['w'] == 'truncnorm':
                    if self.switch == 'sigmoid':
                        deriv_w_norm = self.deriv_w_sigmoid(w)
                    elif self.switch == 'tanh':
                        deriv_w_norm = self.deriv_w_tanh(w)
                    log_prior = self.luniform_ls(w, wa, wb) + \
                        stats.norm.logpdf(w, mean_w, var_w)
                    log_gradient = deriv_w_norm
                elif self.prior_type['w'] == 'free':
                    log_prior = self.luniform_ls(w, wa, wb)
                    log_gradient = 0.0

                return log_prior, log_gradient

            # optimizing changepoint only
            elif cp_opt is True and w_opt is not True:

                cp = np.exp(theta[0])
                cpa = np.exp(self.kernel_.bounds[0, 0])
                cpb = np.exp(self.kernel_.bounds[0, 1])

                # construct the prior and gradient
                if self.prior_type['cp'] == 'truncnorm':
                    deriv_cp_norm = self.deriv_cp(cp)
                    log_prior = self.luniform_ls(cp, cpa, cpb) + \
                        stats.norm.logpdf(cp, 0.98, 0.33)
                    log_gradient = deriv_cp_norm

                elif self.prior_type['cp'] == 'free':
                    log_prior = self.luniform_ls(cp, cpa, cpb)
                    log_gradient = 0.0

                return log_prior, log_gradient

        # stationary kernels start here
        if (self.prior_choice == 'rbfnorm' or self.prior_choice == 'matern32'
                or self.prior_choice == 'matern52'
                or self.prior_choice == 'ratquad'):

            # extract prior params here for easiness of use
            ls_mu = self.prior_params['lengthscale']['mu']
            sig_mu = self.prior_params['sigma']['mu']
            ls_std = self.prior_params['lengthscale']['sig']
            sig_std = self.prior_params['sigma']['sig']

            # begin with sigma
            sig = np.exp(theta[0])
            a_sig = np.exp(self.kernel_.bounds[0, 0])
            b_sig = np.exp(self.kernel_.bounds[0, 1])

            # also load lengthscale
            ls = np.exp(theta[1])
            a = np.exp(self.kernel_.bounds[1, 0])
            b = np.exp(self.kernel_.bounds[1, 1])

            # sigma prior
            log_prior_sig = (self.luniform_sig(sig, a_sig, b_sig)
                             + stats.norm.logpdf(sig, sig_mu, sig_std))
            log_gradient_sig = self.trunc_deriv(sig)

            # now we select the lengthscale prior
            if self.prior_choice == 'rbfnorm':

                def trunc_deriv_rbf(ls):
                    trunc_15 = -(ls - ls_mu)/(ls_std**2)
                    return trunc_15

                log_prior = (self.luniform_ls(ls, a, b)
                             + stats.norm.logpdf(ls, ls_mu, ls_std))
                log_gradient = trunc_deriv_rbf(ls)

            elif self.prior_choice == 'uniform':

                log_prior = self.luniform_ls(ls, a, b)
                log_gradient = self.luniform_ls(ls, a, b)

            elif self.prior_choice == 'matern32':

                def trunc_deriv_matern(ls):
                    trunc_matern = -(ls - ls_mu)/(ls_std**2)
                    return trunc_matern

                log_prior = (self.luniform_ls(ls, a, b)
                             + stats.norm.logpdf(ls, ls_mu, ls_std))
                log_gradient = trunc_deriv_matern(ls)

            elif self.prior_choice == 'matern52':

                def trunc_deriv_matern(ls):
                    trunc_matern = -(ls - ls_mu)/(ls_std**2)
                    return trunc_matern

                log_prior = (self.luniform_ls(ls, a, b)
                             + stats.norm.logpdf(ls, ls_mu, ls_std))
                log_gradient = trunc_deriv_matern(ls)

            elif self.prior_choice == 'ratquad':

                def trunc_deriv_rq(ls):
                    trunc_rq = -(ls - ls_mu)/(ls_std**2)
                    return trunc_rq

                log_prior = (self.luniform_ls(ls, a, b)
                             + stats.norm.logpdf(ls, ls_mu, ls_std))
                log_gradient = trunc_deriv_rq(ls)

            # return both lengthscale and sigma priors together
            return log_prior + log_prior_sig, log_gradient + log_gradient_sig

    # helper pdf functions
    def luniform_ls(self, ls, a, b):
        if ls > a and ls < b:
            return 0.0
        else:
            return -np.inf

    def luniform_sig(self, sig, a, b):
        if sig > a and sig < b:
            return 0.0
        else:
            return -np.inf

    # derivative helper for sigma
    def trunc_deriv(self, sig):
        sig_mu = self.prior_params['sigma']['mu']
        sig_std = self.prior_params['sigma']['sig']
        trunc = -(sig - sig_mu)/(sig_std**2)
        return trunc

    # analytic derivative helper functions
    def deriv_cp(self, cp):
        return -(cp - 0.98)/(0.33**2)

    def deriv_w_sigmoid(self, w):
        return -(w - 0.16)/(0.155**2)

    def deriv_w_tanh(self, w):
        return -(w - 0.32)/(0.31**2)
