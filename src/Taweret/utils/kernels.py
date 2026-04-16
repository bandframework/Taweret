# import external dependences
import numpy as np
from sklearn.gaussian_process.kernels import Kernel, RBF, Hyperparameter
from sklearn.gaussian_process.kernels import ConstantKernel as C


# set up a wrapper for sklearn kernel class
class SigmoidChangepoint(Kernel):

    r'''
    Designs a non-stationary changepoint kernel that inherits
    from the sklearn RBF Kernel class.

    The kernel is given by:
    .. math::
        k(x_i, x_j) = (1 - \\sigma(x_i)) * K1(x_i,x_j) *
                      (1 - \\sigma(x_j)) + \\sigma(x_i) * K2(x_i,x_j)
                      * \\sigma(x_j)

    where K1 is the first kernel and K2 is the second kernel, with the
    changepoint defined by a chosen switching function. The current
    options are 'linear' and 'sigmoid'.
    '''

    # import hyperparameter names here
    def __init__(self, ls1, ls2, cbar1, cbar2, changepoint=1.0,
                 changepoint_bounds=(1e-5, 1e5),
                 width=1.0, width_bounds=(1e-5, 1e5)):

        # will not touch these for now
        self.ls1 = ls1
        self.ls2 = ls2
        self.cbar1 = cbar1
        self.cbar2 = cbar2

        # optimizable parameters
        self.width = width
        self.width_bounds = width_bounds
        self.changepoint = changepoint
        self.changepoint_bounds = changepoint_bounds

        # which kernel type am I using
        self.type = 'sigmoid'

        return None

    @property
    def anisotropic(self):
        return np.iterable(self.changepoint) and len(self.changepoint) > 1

    # get hyperparmeters to be optimized, leave option for fixed values
    @property
    def hyperparameter_changepoint(self):
        if self.anisotropic:
            return Hyperparameter(
                "changepoint",
                "numeric",
                self.changepoint_bounds,
                len(self.changepoint),
            )
        return Hyperparameter("changepoint", "numeric",
                              self.changepoint_bounds)

    @property
    def hyperparameter_width(self):
        if self.anisotropic:
            return Hyperparameter(
                "width",
                "numeric",
                self.width_bounds,
                len(self.width),
            )
        return Hyperparameter("width", "numeric", self.width_bounds)

    # call the function, see what happens
    def __call__(self, X, Y=None, eval_gradient=False):

        # check the dimensions
        X = np.atleast_2d(X)

        # this should work for all kernels
        if Y is None:
            Y = X

        # initialize the K kernel matrix (len(tr_data), len(tr_data))
        self.K = np.zeros([len(X), len(Y)])

        # assign the stationary kernels (chiral and pQCD)
        self.K1 = (C(constant_value=self.cbar1, constant_value_bounds='fixed')
                   * RBF(length_scale=self.ls1,
                         length_scale_bounds='fixed'))(X, Y)

        self.K2 = (C(constant_value=self.cbar2, constant_value_bounds='fixed')
                   * RBF(length_scale=self.ls2,
                         length_scale_bounds='fixed'))(X, Y)

        self.K3 = C(constant_value=0.0, constant_value_bounds='fixed')(X, Y)

        # if statement for cases
        if self.type == 'theta':

            # assign Heaviside functions like sigmoid below
            h1 = np.heaviside(X - self.changepoint, 1).T
            h2 = np.heaviside(Y - self.changepoint, 1).T
            self.K = np.outer(np.ones(len(X)) - h1,
                              np.ones(len(Y)) - h2) * self.K1
            + np.outer(h1, h2) * self.K2
            + np.outer(np.ones(len(X)) - h1, h2) * self.K3
            + np.outer(h1, np.ones(len(Y)) - h2) * self.K3

        elif self.type == 'sigmoid':

            # sigmoid bilinear function
            def sigmoid(dens, x0, k):
                return 1.0 / (1.0 + np.exp(-(dens-x0)/k))

            # sigmoid deriv for cp and width
            def sig_grad(dens, x0, k, deriv='cp'):
                if deriv == 'cp':
                    grad = (-(np.exp(-(dens-x0)/k)/k)
                            * (1.0 + np.exp(-(dens-x0)/k))**(-2.0))
                    return grad
                elif deriv == 'w':
                    grad = (-(np.exp(-(dens-x0)/k))
                            * (dens-x0)/k**2.0
                            * (1.0 + np.exp(-(dens-x0)/k))**(-2.0))
                    return grad

            # define kernel matrix
            self.K = (np.outer((np.ones(len(X))
                               - sigmoid(X, self.changepoint, self.width).T),
                               (np.ones(len(Y))
                               - sigmoid(Y, self.changepoint, self.width).T))
                      * self.K1)
            + np.outer(sigmoid(X, self.changepoint, self.width).T,
                       sigmoid(Y, self.changepoint, self.width).T) * self.K2

        # only for when optimization of hyperparameters is needed
        if eval_gradient:

            if self.anisotropic is False:

                # go into the function type
                if self.type == 'sigmoid':

                    # gradient wrt changepoint
                    if self.hyperparameter_changepoint.fixed is False:
                        sx = sigmoid(X, self.changepoint, self.width).T
                        sy = sigmoid(Y, self.changepoint, self.width).T
                        gx = sig_grad(X, self.changepoint, self.width).T
                        gy = sig_grad(Y, self.changepoint, self.width).T

                        onex = np.ones(len(X))
                        oney = np.ones(len(Y))

                        K_gradient_cp = (self.changepoint * (
                                             np.outer(-gx, (oney - sy))
                                             * self.K1
                                             + np.outer((onex - sx), -gy)
                                             * self.K1
                                             + np.outer(gx, sy)
                                             * self.K2
                                             + np.outer(sx, gy)
                                             * self.K2))[:, :, np.newaxis]
                    else:
                        K_gradient_cp = np.empty((self.K.shape[0],
                                                  self.K.shape[1], 0))

                    # gradient wrt width
                    if self.hyperparameter_width.fixed is False:
                        sx = sigmoid(X, self.changepoint, self.width).T
                        sy = sigmoid(Y, self.changepoint, self.width).T
                        gx = sig_grad(X, self.changepoint, self.width, 'w').T
                        gy = sig_grad(Y, self.changepoint, self.width, 'w').T

                        onex = np.ones(len(X))
                        oney = np.ones(len(Y))

                        self.K_gradient_w = (self.width
                                             * (np.outer(-gx, (oney - sy))
                                                 * self.K1
                                                 + np.outer((onex - sx), -gy)
                                                 * self.K1 + np.outer(gx, sy)
                                                 * self.K2
                                                 + np.outer(sx, gy)
                                                 * self.K2))[:, :, np.newaxis]
                    else:
                        self.K_gradient_w = np.empty((self.K.shape[0],
                                                      self.K.shape[1], 0))

                    # full gradient return
                    stack_grad = np.dstack((K_gradient_cp, self.K_gradient_w))
                    return self.K, stack_grad

                elif self.type == 'theta':
                    raise ValueError("""The gradient cannot be evaluated
                                     for the Heaviside changepoint kernel."""
                                     """Optimization cannot be performed
                                     using this kernel choice.""")

            elif self.anisotropic:
                raise ValueError("""The kernel has not been implemented
                                 for anisotropic cases.""")

        # if eval_gradient is false, return no gradient
        else:
            return self.K

    # diagonal function (general for the nonstationary case)
    def diag(self, X):
        return np.apply_along_axis(self, 1, X).ravel()

    # stationary function needed for base class
    def is_stationary(self):
        return False

    # this returns the parameter value when trained
    def __repr__(self):
        if self.anisotropic:
            return "{0}(changepoint=[{1}])".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.changepoint)),
            )
        else:  # isotropic
            return "{0}(changepoint={1:.3g}, width={2:.3g})".format(
                self.__class__.__name__, self.changepoint, self.width
            )


class TanhChangepoint(Kernel):

    r'''
    Designs a non-stationary changepoint kernel that inherits
    from the sklearn RBF Kernel class.

    The kernel is given by:
    .. math::
        k(x_i, x_j) = (1 - \\sigma(x_i)) * K1(x_i,x_j) *
                      (1 - \\sigma(x_j)) + \\sigma(x_i) * K2(x_i,x_j)
                      * \\sigma(x_j)

    where K1 is the first kernel and K2 is the second kernel, with the
    changepoint defined by a chosen switching function. The current
    option is 'tanh'.
    '''

    # import hyperparameter names here
    def __init__(self, ls1, ls2, cbar1, cbar2, changepoint=1.0,
                 changepoint_bounds=(1e-5, 1e5),
                 width=1.0, width_bounds=(1e-5, 1e5)):

        # will not touch these for now
        self.ls1 = ls1
        self.ls2 = ls2
        self.cbar1 = cbar1
        self.cbar2 = cbar2

        # optimizable parameters
        self.width = width
        self.width_bounds = width_bounds
        self.changepoint = changepoint
        self.changepoint_bounds = changepoint_bounds

        return None

    @property
    def anisotropic(self):
        return np.iterable(self.changepoint) and len(self.changepoint) > 1

    # get hyperparmeters to be optimized, leave option for fixed values
    @property
    def hyperparameter_changepoint(self):
        if self.anisotropic:
            return Hyperparameter(
                "changepoint",
                "numeric",
                self.changepoint_bounds,
                len(self.changepoint),
            )
        return Hyperparameter("changepoint", "numeric",
                              self.changepoint_bounds)

    @property
    def hyperparameter_width(self):
        if self.anisotropic:
            return Hyperparameter(
                "width",
                "numeric",
                self.width_bounds,
                len(self.width),
            )
        return Hyperparameter("width", "numeric", self.width_bounds)

    # call the function, see what happens
    def __call__(self, X, Y=None, eval_gradient=False):

        # check the dimensions
        X = np.atleast_2d(X)

        # this should work for all kernels
        if Y is None:
            Y = X

        # initialize the K kernel matrix (len(tr_data), len(tr_data))
        self.K = np.zeros([len(X), len(Y)])

        # assign the stationary kernels (chiral and pQCD)
        self.K1 = (C(constant_value=self.cbar1, constant_value_bounds='fixed')
                   * RBF(length_scale=self.ls1,
                         length_scale_bounds='fixed'))(X, Y)

        self.K2 = (C(constant_value=self.cbar2, constant_value_bounds='fixed')
                   * RBF(length_scale=self.ls2,
                         length_scale_bounds='fixed'))(X, Y)

        self.K3 = C(constant_value=0.0, constant_value_bounds='fixed')(X, Y)

        # tanh function
        def tanh(dens, x0, k):
            return 0.5 + 0.5 * np.tanh((dens - x0)/k)

        # tanh deriv for cp and width
        def tanh_grad(dens, x0, k, deriv='cp'):
            if deriv == 'cp':
                grad = -0.5 * np.cosh((dens - x0)/k)**(-2.0) / k
                return grad
            elif deriv == 'w':
                grad = (-0.5 * (dens - x0) *
                        np.cosh((dens - x0)/k)**(-2.0) / k**2.0)
                return grad

        # define kernel matrix
        self.K = np.outer((np.ones(len(X))
                           - tanh(X, self.changepoint, self.width).T),
                          (np.ones(len(Y))
                           - tanh(Y, self.changepoint, self.width).T)) \
            * self.K1 + (np.outer(tanh(X, self.changepoint, self.width).T,
                                  tanh(Y, self.changepoint, self.width).T)
                         * self.K2)

        # only for when optimization of hyperparameters is needed
        if eval_gradient:

            if self.anisotropic is False:

                # gradient wrt changepoint
                if self.hyperparameter_changepoint.fixed is False:
                    tx = tanh(X, self.changepoint, self.width).T
                    ty = tanh(self.changepoint, self.width, Y).T
                    gx = tanh_grad(X, self.changepoint, self.width).T
                    gy = tanh_grad(Y, self.changepoint, self.width).T

                    onex = np.ones(len(X))
                    oney = np.ones(len(Y))

                    K_gradient_cp = (self.changepoint
                                     * (np.outer(-gx, (oney - ty))
                                         * self.K1 + np.outer((onex - tx), -gy)
                                         * self.K1 + np.outer(gx, ty)
                                         * self.K2 + np.outer(tx, gy)
                                         * self.K2))[:, :, np.newaxis]
                else:
                    K_gradient_cp = np.empty((self.K.shape[0],
                                              self.K.shape[1], 0))

                # gradient wrt width
                if self.hyperparameter_width.fixed is False:
                    tx = tanh(X, self.changepoint, self.width).T
                    ty = tanh(Y, self.changepoint, self.width).T
                    gx = tanh_grad(X, self.changepoint, self.width, 'w').T
                    gy = tanh_grad(Y, self.changepoint, self.width, 'w').T

                    onex = np.ones(len(X))
                    oney = np.ones(len(Y))

                    self.K_gradient_w = (self.width
                                         * (np.outer(-gx, (oney - ty))
                                             * self.K1
                                             + np.outer((onex - tx), -gy)
                                             * self.K1 + np.outer(gx, ty)
                                             * self.K2 + np.outer(tx, gy)
                                             * self.K2))[:, :, np.newaxis]
                else:
                    self.K_gradient_w = np.empty((self.K.shape[0],
                                                  self.K.shape[1], 0))

                # full gradient return
                return self.K, np.dstack((K_gradient_cp, self.K_gradient_w))

            elif self.anisotropic:
                raise ValueError("""The kernel has not been implemented
                                 for anisotropic cases.""")

        # if eval_gradient is false, return no gradient
        else:
            return self.K

    # diagonal function (general for the nonstationary case)
    def diag(self, X):
        return np.apply_along_axis(self, 1, X).ravel()

    # stationary function needed for base class
    def is_stationary(self):
        return False

    # this returns the parameter value when trained
    def __repr__(self):
        if self.anisotropic:
            return "{0}(changepoint=[{1}])".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.changepoint)),
            )
        else:  # isotropic
            return "{0}(changepoint={1:.3g}, width={2:.3g})".format(
                self.__class__.__name__, self.changepoint, self.width
            )
