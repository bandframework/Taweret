# Bivariate linear model mixing: an introduction

Bivariate linear model mixing is able to combine two independent models using either the density- or mean-mixing strategies discussed in the previous chapter. The posterior predictive distribution can be written generally, for the density-based mixing, as

$$
 p(F(x)|\theta) \sim \alpha(x;\theta) F_1(x) +  (1-\alpha(x;\theta)) F_2(x),
$$

where $F(x)$ is the underlying theory we wish to describe with the combination of the individual models, $F_1(x)$ and $F_2(x)$. Here, $\alpha(x, \theta)$ is the mixing function, dependent on input space $x$ and hyperparameters $\theta$, chosen by the user to combine the two models. This choice should be informed by the understanding of the system at hand, and hence influences the result of the model mixing considerably. The current possible choices for this mixing function are

- Step function: $\Theta(\beta_{0} - x)$;
- Asymmetric 2-step: $\zeta \Theta(\beta_{0} -x) + (1 - \zeta) \Theta(\beta_{0} -x)$;
- Sigmoid: $\exp((x - \beta_{0})/\beta_{1})$;
- Cosine: 

$$
\alpha(x; \theta) = 
    \begin{cases} 
        1, & x \leq \theta_{1}; \\
        \frac{1}{2}\left[1 + \cos(\frac{\pi}{2} \left(\frac{x-\theta_{1}}{\theta_{2} - \theta_{1}}\right))\right], & \theta_{1} < x \leq \theta_{2}; \\
        \frac{1}{2}\left[1 + \cos\left(\frac{\pi}{2} \left(1 + \frac{x - \theta_{2}}{\theta_{3} - \theta_{2}} \right) \right) \right],  & \theta_{2} < x \leq \theta_{3}; \\
        0, & x > \theta_{3}.
    \end{cases}
$$

In all of the above functions, $\beta_{0}, \beta_{1}$ correspond to the shape parameters of the mixing function, $\zeta$ is a mixing hyperparameter, and $\theta_{i}$ are also mixing function hyperparameters.

:::{important}
In this method, the models are expected to be one-dimensonally mixed, but there is currently capability to simultaneously mix the models and determine the hyperparameters of each model (calibration), which is not a feature built into any of the other model mixing methods included in this package at the present time. Future work will look to include this simultaneous calibration and mixing a reality in all model mixing scenarios.
:::