Taweret for Bayesian Model Mixing
=================================

.. contents::
    :local:

Bayesian Model Mixing
---------------------

In computer simulation studies, it is often the case where a set of theoretically \
sound models are considered, each intended to describe the true underlying physical phenomena of interest \
across a sub-region of the domain. These models could differ by their underlying assumptions \
and may not be valid in certain sub-regions of the input domain. In practice, the true underlying \
model is not contained in the set of candidate models. Thus, selecting a single model to describe the true phenomena \
across the entire input domain is inappropriate. As an alternative, one may elect to combine the information within \
the model set in some systematic manner. A common approach involves combining the individual \
mean predictions or predictive densities from the indivdual models using a linear combination or weighted average. \
The weights in this linear combination may or may not depend on the inputs. When the models under consideration \
exhibit varrying levels of predictive accuracy depending on the sub-region of the input domain, an input-dependent \
weighting scheme is more appropriate. A memeber of the class of input-dependent weighting schemes is \
Bayesian Model Mixing (BMM). BMM is a data-driven technique which combines the predictions from a set of N candidate models in a \
Bayesian manner using input-dependent weights. Mixing can be performed using one of two strategies described below: \
(1) A two-step approach: Each model is fit prior to mixing. \
The weight functions are then learned conditional on the predictions from each model. \
(2) A joint analysis: When the models have unknown parameters, one could elect to perform calibration while simultaneously \
learning the weight functions.   

Taweret is a python package which provides a variety of BMM methods. Each method combines the information across a set of N models \
in a Bayesian manner using an input-dependent weighting scheme. The BMM methods in Taweret are designed to esitmate the \
true mean of the underlying system (mean-mixing) or the true predictive density of the underlying system (density-mixing). \
Selecting a mixing objectve (mean vs. density mixing) and associated method is problem dependent.  

The typical workflow of Bayesian Model Mixing includes:

1. Define a set of candidate models and collect experimental data. 
2. Decide on a mixing method.
3. Learn the weights/mixing function to combine models.

Taweret for Bayesian Model Mixing
---------------------------------

.. image:: _static/Taweret.png

Taweret is the protective ancient Egyptian goddess of childbirth and fertility. She has the head of a hippopotamus \
and limbs and paws of a lion. Her back and tail is of a Nile crocodile. Hence the name of our Bayesian Model \
Mixing package, Taweret!


Models
^^^^^^
The user has to provide models that they would like to mix. Currently Taweret supports mixing of two \
or more models with a 1,...,p-dimensional input space (depending on the method of mixing chosen) and a single output. \
The models are required to have an "evaluate" a method which should return a mean and a standard deviation for each input parameter value. 

Mixing Method
^^^^^^^^^^^^^
The user will then choose a mixing method. Currently Taweret supports: \
1. **Linear mixing** (2 models) \
2. **Multivariate BMM** (N models) \
3. **Bayesian Trees** (N models) \

Details of each Mixing Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Linear mixing**: A density-mixing method employing a mixing function between the two models chosen \
with four different types of mixing functions available in Taweret (*Step*, *Sigmoid*, \
*CDF*, *Piecewise cosine*). This mixing method takes two models and the experimental data as input \
and calculates the likelihood. Finding of the optimal weights by either optimizing the \
likelihood or finding the full posterior is done as the next step. 

**Multivariate BMM**: A mean-mixing method that combines two (or more!) models provided by the user into \
a mixed model. This method, unlike linear mixing, only requires knowledge of the two models and their \
uncertainties at the input points. Given that the models are supplied by the user, one could combine \
two functions with a Gaussian Process (as seen in the example notebook for this method). 

**Bayesian Trees**: This mean-mixing method estimates the true underlying system by combining the mean predictions \
from N models using a linear combination and input-dependent weighting scheme. The weights functions \
are defined using Bayesian Additive Regression Trees (BART). This flexible and non-parametric weighting scheme \
allows the weight functions to reflect the localized performances of each model based on the information across \
a set of observational data and the corresponding mean predictions from the model set. This approach is applicable for \
p-dimensional input spaces.     

Estimating the Weight Functions 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Taweret provides a variety of BMM methods, each which utilize an input-dependent weighting scheme. \
The weighting scheme may vary substantially across the different methods. For example, Linear mixing \
defines the weights using a parametric model, while the Bayesian Trees approach uses a non-parametric model. \
Another weighting scheme involves precision weighting, as seen in Multivariate BMM. Hence, the exact estimation \
of the weight functions may differ substantially across the various BMM methods. Despite this, the estimation \
process in each method is facilitated using Bayesian principles. Examples of each method can be found in the \
Python notebooks (docs/source/notebooks) and under the Examples tab on this page. In these examples, BMM is \
applied to the SAMBA, Coleman, and Polynomial models.

Working with Multiple Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**A Two-step approach**: \
In some cases, the models under consideration may have been previously calibrated. \
Consequently, the predictions from each model are easily ascertained across a new set of input locations. This calibration \
phase is the first step in the two-step process. The second step invloves mixing the predictions from each model \
to estimate the true system. Thus, conditional on the individual predictions across a set of inputs along with observational data, \
the weight functions are learned and the overall mean or predictive density of the underlying system is estimated in a Bayesian manner. \
Examples of this two-step analysis can be found in a variety of the notebooks provided in the Examples section.  


**Mixing and Calibration**: \

This joint analysis is advantageous because it enables each model to be calibrated predominantly based on the sub-regions \
of the domain where its predictions align well with the observational data. These sub-regions will be simultaneously identified \
by the weight functions. This should lead more reliable inference than then case where each model is calibrated individually and \
thus forced to reflect a global fit to the data. For example, the joint analysis would avoid situations where a model is calibrated \
using experimental data that is outside its applicability. Examples of this joint analysis are applied to the Coleman models.   
