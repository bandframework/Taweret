# Multivariate Bayesian model mixing: an introduction

Multivariate model mixing is designed to use the precision of the model at each point in the input space to weight each individual model locally. This method does not require the definition of a mixing function, since the weights are pre-determined by the model variances at each point in the space, eliminating the bias of choosing a functional form for the mixing from the result. This method is purely moment-based mixing, or mean-mixing, and can be described as

$$
\mathcal{M}_{\dagger} \sim \mathcal{N}(f_{\dagger}, Z_{P}^{-1}): \quad f_{\dagger} = \frac{1}{Z_{P}} \sum_{k=1}^{K} \frac{1}{\sigma_{k}^{2}} f_{k}, \quad Z_{P} \equiv \sum_{k=1}^{K} \frac{1}{\sigma_{k}^{2}},
$$

hence

$$
\mathcal{M}_{k} \sim \mathcal{N}(f_k(x), \sigma_{k}^{2}(x)),
$$

where $Z_{P}$ is the precision of the models, or the inverse of the variances. $f_{\dagger}$ is the desired mean result of the mixed model, $f_{k}(x)$ the mean of the $k$th model, and $\sigma_{k}^{2}$ the variance of the $k$th model at each point in $x$. 

This method is also one-dimensional at present, and requires the full PPD of the models it is mixing, but future developments hope to include simultaneous model calibration and mixing. See [this work](https://doi.org/10.1103/PhysRevC.106.044002) for more details of using this package on a toy model for effective field theories (EFTs), and [this package](https://github.com/asemposki/SAMBA) for the original code containing this mixing method.