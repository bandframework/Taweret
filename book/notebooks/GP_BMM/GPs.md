# Model mixing with Gaussian processes

In this section, we will briefly describe how to use Gaussian processes (GPs), and how they can be used for Bayesian model mixing (BMM). We overview the basic form of a GP, what a 'stationary' or 'nonstationary' kernel is, and how to constrain the parameters of the GP according to our chosen models. We then outline the procedure for using GPs to perform BMM, which we extract from our paper found [here](https://arxiv.org/abs/2404.06323).

## What is a Gaussian process?

According to [Rasmussen and Williams](https://gaussianprocess.org/gpml/chapters/RW.pdf), a GP is simply "a collection of random variables, any finite number of which have a joint Gaussian distribution." This is rather vague, however, so we will instead use a more intuitive picture: a GP is a distribution built from a family of basis functions that is able to predict the mean and covariance of unknown data points given known data. It is defined by the functional form

$$
f(x) \sim \mathcal{GP}(m(x), \kappa(x,x')),
$$

where $m(x)$ is the mean function of the GP and $\kappa(x,x')$ is the covariance function, or kernel. The mean function captures the overall trend of the data, and the covariance function is meant to describe the correlations between the data points, and the deviation of the data from this overall mean. For the cases we are studying, we will be setting the mean function to zero, but a user can change this option themselves to whatever mean function they wish to use. In terms of the kernel, a popular choice is the stationary, squared-exponential radial basis function (RBF)

$$
\kappa(x,x';\ell) = \exp \left(\frac{-(x-x')^2}{2\ell^2}\right),
$$

where $\ell$ is the lengthscale of the kernel, and represents the length of correlations in the data across the input space. In `Taweret`, we will access this kernel, along with the Matérn and rational quadratic kernels, both of which are stationary kernels with different functional forms.

## What is 'stationary' vs. 'nonstationary'?

You may be wondering what the word 'stationary' really means here; this is simply denoting that the kernel depends only on the Euclidean distance between the data in the input space, i.e., $|x-x'|$. Conversely, 'nonstationary' means that the kernel depends on the location of the data points in the input space, i.e., $x$. These kernels are especially useful for physics problems, since data can have more structure than can be described with a stationary covariance function. In `Taweret`, a nonstationary, changepoint kernel is supplied for these cases, and users can develop their own nonstationary kernels based on its code structure.

## Constraining the kernel parameters

All GP kernels possess at least one hyperparameter to be learned from the data. In our case, we are using model information as data, so we will need to constrain our GP by training on this 'model data'. However, standard GPs use the maximum likelihood estimation (MLE) procedure to constrain these parameters, which can in turn lead to correlation lengths that are far too long for the given problem to properly describe the physical system at hand. This conundrum leads to the question: can we instead build in physical properties of our system through constraints on the hyperparameters? The answer, of course, is yes we can! In `Taweret`, we are able to do this by implementing hyperpriors on these parameters and employing a maximum a posteriori (MAP) procedure to obtain the most probable value of each kernel hyperparameter. The `GPPriors` class contains standard truncated normal distributions for the hyperparameters in the supplied kernels; users can also implement their own prior forms if they would like to by following the `GPPriors` class as a template.

## The model mixing approach

Let's now move to how the GP can perform the model mixing. Let's say we have two models, Model A and Model B, which both have the form

$$
Y^{(i)}(x) = F(x) + \delta Y^{(i)}(x), \qquad i \in [1, M],
$$

where $Y^{(i)}(x)$ is a random variable that represents predictions of Model $i$, here A or B, at some points $x$. $\delta Y^{(i)}(x)$ is the error of the model, and $F(x)$ is the underlying theory we're trying to represent with the models we have defined. If we then say that the data possesses covariances from this error that is included in the model, we can write the covariances as 

$$
\kappa^{(i)}_y(x,x'),
$$

where again there will be one of these matrices per model $i$. Usually, the covariances can be written as a distribution themselves---in our examples, they will be modelled using Gaussian distributions. Hence, the models and their uncertainties are probability distributions that we can denote as

$$
p(y^{(i)}(x)|f(x), \kappa^{(i)}_y),
$$

and we will use these in Bayes' theorem to determine a common mean function $F(x)$ by combining the distributions. To perform that calculation, we need to define a prior on the underlying theory---this is where the GP comes in:

$$
F(x) \sim \textrm{GP}[0, \kappa_{f}(x,x')].
$$

Hence, we train the GP with information from the two models and the GP combines this information into the mixed model we are looking for. This yields a non-local mixing strategy. 

We therefore sample data points from the two models that possess these covariances, and combine the two sets of training data to give to the GP. Our input array of training locations is called $\vec{x}_t$, and our observed training data is denoted as $\vec{y}_t$. To include correlations between data points in each model, we form a block-diagonal covariance matrix from their individual covariances:

$$
K_y = \kappa^{(A)}(\vec{x}_{t,A},\vec{x}_{t,A}) \otimes \kappa^{(B)}(\vec{x}_{t,B}, \vec{x}_{t,B}).
$$

This allows us to keep the models uncorrelated but include the intra-model correlations. This is added to the GP's covariance matrix coming from its kernel in the training step, which we write as $K_f = \kappa_f(x,x')$. Now we want to evaluate at another set of points, so we will denote these as $\vec{x}_e$. We can, after some algebra, write the log posterior, up to constants, of the underlying theory $\vec{f}$ at both training and prediction points as

$$
\text{log} p(\vec{f} | \vec{y}, K_y, K_f) = -\frac{1}{2} (\vec{f} - \vec{\mu})^T \Sigma^{-1} (\vec{f} - \vec{\mu}) + \ldots.
$$

And hence $\vec{f}$ is a multivariate Gaussian that can be defined as

$$
\vec{F} | \vec{y}, K_y, K_f \sim \mathcal{N}[\vec{\mu}, \Sigma],
$$

where

$$
\vec{\mu} \equiv \Sigma B_t^T K_y^{-1} \vec{y},
$$

where $B_t$ selects for the training points out of the full $\vec{f}$ array, and

$$
\Sigma \equiv (K_f^{-1} + B_t^T K_y^{-1}B_t)^{-1}.
$$

To obtain the final result for $\vec{F}$ at any evaluation points, we use $B_e$, which selects now for the evaluation points instead of the training points, and obtain

$$
\vec{\mu}_e = K_{f, et} (K_{f, tt} + K_{y, tt})^{-1} \vec{y}, \nonumber \\
\Sigma_{ee} = K_{f, ee} - K_{f, et} (K_{f, tt} + K_{y, tt})^{-1} K_{f, te}.
$$

These are the results of the GP training, and will allow us to make predictions of $\vec{F}$ at any locations we desire in the input space. Hence, we have produced the model-mixed result!

## Up next

In our next two tutorials, we will implement Bayesian model mixing using GPs on the toy SAMBA models we used for the multivariate mixing and linear model mixing methods earlier in this Book. We will also highlight the possible choices of stationary and nonstationary kernels available in Taweret. We will outline how to select hyperpriors for the kernel hyperparameters, how to set up the GP code, and how to analyze the results. We hope you will learn how to use GPs for model mixing and will be able to extend these techniques to your own research!

:::{seealso}
See [this thesis](http://rave.ohiolink.edu/etdc/view?acc_num=ohiou1752925796224309) for more extensive details on GPs and the model mixing techniques found in this chapter. 
:::
