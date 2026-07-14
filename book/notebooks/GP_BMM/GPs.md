# Model mixing with Gaussian processes

In this section, we will briefly describe how to use Gaussian processes (GPs), and how they can be used for Bayesian model mixing (BMM). We overview the basic form of a GP, what a 'stationary' or 'nonstationary' kernel is, and how to constrain the parameters of the GP according to our chosen models. 

## What is a Gaussian process?

Put simply, a GP is 'a collection of random variables, any finite number of which have a joint Gaussian distribution."[(see this reference)](https://gaussianprocess.org/gpml/chapters/RW.pdf). This is rather vague, however, so we will instead use a more intuitive picture: a GP is a distribution built from a family of basis functions that is able to predict the mean and covariance of unknown data points given known data. It is defined by the functional form

$$
f(x) \sim \mathcal{GP}(m(x), \kappa(x,x')),
$$

where $m(x)$ is the mean function of the GP and $\kappa(x,x')$ is the covariance function, or kernel. The mean function captures the overall trend of the data, and the covariance function is meant to describe the correlations between the data points, and the deviation of the data from this overall mean. For the cases we are studying, we will be setting the mean function to zero, but a user can change this option themselves to whatever mean function they wish to use. In terms of the kernel, a popular choice is the stationary, squared-exponential radial basis function (RBF)

$$
\kappa(x,x';\ell) = \exp(\frac{-(x-x')}{2\ell}),
$$

where $\ell$ is the lengthscale of the kernel, and represents the length of correlations in the data across the input space. In `Taweret`, we will access to this kernel, along with the Mat\'ern and rational quadratic, both of which are stationary kernels with different functional forms. 

## What is 'stationary' vs. 'nonstationary'?

You may be wondering what the word 'stationary' really means here; this is simply denoting that the kernel depends only on the Euclidean distance between the data in the input space, i.e., $(x-x')$. Conversely, 'nonstationary' means that the kernel depends on the location of the data points in the input space, i.e., $x$. These kernels are especially useful for physics problems, since data can have more structure than can be described with a stationary covariance function. In `Taweret`, a nonstationary, changepoint kernel is supplied for these cases, and users can develop their own nonstationary kernels based on its code structure.

## Constraining the kernel parameters

All GP kernels possess at least one hyperparameter to be learned from the data. In our case, we are using model information as data, so we will need to constrain our GP by training on this 'model data'. However, standard GPs will use the maximum likelihood estimation (MLE) procedure to constrain these parameters, which can in turn lead to correlation lengths that are far too long for the given problem to properly describe the physical system at hand. This conundrum leads to the question: can we instead build in physical properties of our system through constraints on the hyperparameters? The answer, of course, is yes we can! In `Taweret`, we are able to do this by implementing hyperpriors on these parameters and employing a maximum a posteriori (MAP) procedure to obtain the most probable value of each kernel hyperparameter. The `GPPriors` class contains standard truncated normal distributions for the hyperparameters in the supplied kernels; users can also implement their own prior forms if they would like to by following the `GPPriors` class as a template.

## Up next

In our next two tutorials, we will implement the stationary and nonstationary kernels on the toy SAMBA models we used for the multivariate mixing and linear model mixing methods earlier in this Book. We will outline how to select hyperpriors for the kernel hyperparameters, how to set up the GP code, and how to analyze the results. We hope you will learn how to use GPs for model mixing and will be able to extend these techniques to your own research!

:::{seealso}
See [this thesis](https://etd.ohiolink.edu/acprod/odb_etd/r/etd/search/10?p10_accession_num=ohiou1752925796224309&clear=10&session=3431789771542) for more extensive details on GPs and the model mixing techniques found in this chapter. 
:::