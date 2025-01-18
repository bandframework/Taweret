# Bayesian model mixing with regression trees: an introduction

The third mixing method in Taweret is the combination of models using decision trees to determine the weights of each. This allows for adaptive learning of the weights, and again eliminates the bias of choosing a mixing function form. The weights can be described as

$$
w_{k}(x) = \sum_{j=1}^{m} g_{k}(x;T_{j};M_{j}), \quad \textrm{for}~~~k = 1, \dots, K.
$$

Here, $g_{k}(x;T_{j};M_{j})$ is the $k$th output of the $j$th tree, given by $T_{j}$. The set of parameters associated with this tree is $M_{j}$. The weights are normalized to a prior interval $[0,1]$, but this condition is not strictly enforced, allowing for values outside of this interval.

This mixing method interfaces with the BART (Bayesian Additive Regression Trees) C++ package, [`openBT`](https://bitbucket.org/mpratola/openbt/wiki/Home), and is able to handle multi-dimensional model mixing. 

:::{seealso}
See [this paper](https://doi.org/10.1080/00401706.2023.2257765) for more details. 
:::