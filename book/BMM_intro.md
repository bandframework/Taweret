# Bayesian Model Mixing: a short introduction to the field

Uncertainty quantification using Bayesian methods is a still at the frontiers of modern research, and Bayesian model mixing is one of the branches of this field. Bayesian model mixing (BMM) combines the predictions from multiple models such that the fidelity of each model is preserved in the final, mixed result. However, as of yet, no comprehensive review article or package exists to facilitate easy access to both the concepts and numerics of this research area. This package helps to satisfy the latter need, containing state of the art techniques to perform Bayesian model mixing on many types of scenarios provided by the user, as well as some toy models already encoded for practise with these methods. Currently, Taweret contains three BMM techniques, each pertaining to a different general problem, and has been designed as a conveniently modular package so that a user can add their own model mixing method or model to the framework. Taweret is also generally formulated so that it can be used in more than just the nuclear physics branch of science in which it was originally tested.

## What is Bayesian model mixing?

Bayesian model mixing attempts to combine various models that describe the same underlying system in several different ways, the appropriate one for the scenario decided by the specifics of the problem and the information available. The overarching theme of BMM, however, is that it uses input-dependent weights to combine the models at hand---a technique different from the more well-known Bayesian model averaging (BMA). The latter technique often uses the model evidences as global weights to average the models over the input space. It is arguable that BMM is an improvement over BMA due to its dependence on location in the input space, allowing more precise model weighting and maximizing the predictive power of each individual model over the region in which it dominates.

Two common groups used to describe Bayesian model mixing are called "mean mixing" and "density mixing"---the former involving a weighting of the moments of two or more models, and the latter a mixing of the entire posterior distribution of each model. Mean mixing can be summarized by

$$
E[Y | x] = \sum_{k=1}^{K} w_{k}(x) f_{k}(x), 
$$

where $E[Y | x]$ is the mean of $Y$ observations given the input parameter vector $x$. $f_{k}(x)$ is the mean prediction of each $k$th model $M_{k}$, and $w_{k}(x)$ are the input-dependent weights of each model. Density mixing can be written as

$$
p[Y_{0} | x_{0}, Y] = \sum_{k=1}^{K} w_{k}(x_{0}) p(Y_{0} | x_{0}, Y, M_{k}), 
$$

where $p(Y_{0} | x_{0}, Y, M_{k})$ is the predictive density of a future observation $Y_{0}$ given the location $x_{0}$ and the $k$th model $M_{k}$.

It can be easily seen that the weight function is the most difficult decision to make in each individual problem encountered. In Taweret, weight functions are defined for each method, as users will see in the package structure and in the following tutorials. 

## Outline of this Book

In the subsequent chapters, there are numerous tutorials to help a new user get comfortable with the model mixing methods in this package, and to test each to determine which one is best for their use case. We begin with linear model mixing, which works best when experimental data is available to help train hyperparameters of the mixing function chosen. We then move to multivariate model mixing, where each model is formulated as a Gaussian distribution, and precision-weighting is used to combine the models across the input space.

We encourage users to play with the codes and improve them, or implement their own model mixing methods and toy models in the future. As the field of Bayesian model mixing expands, Taweret should expand with it to contain as many techniques as possible so that users have a single package to turn to when they desire to try out different model mixing on their problems, whether they be in nuclear physics or meteorology, statistics, or any other field requiring input from Bayesian inference.