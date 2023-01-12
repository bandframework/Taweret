Taweret for Bayesian Model Mixing
=================================

.. contents::
    :local:

Bayesian Model Mixing
---------------------

In computer simulation studies we often come across multiple theoretically \
sound models that are equally good. These models have different assumptions built into them \
and are only valid in certain regions of the input domain. It is most certainly possible that the true \
model that can describe the phenomena is not any of the models that are being considered. Bayesian Model \
Mixing is a data-driven technique that allows us to combine the models to get the most accurate predictions\
and calibrations while taking into account the modeling uncertainties. Taweret is a python package \
for Bayesian Model Mixing. 

Typical workflow of Bayesian Model Mixing

1. Collect all the models and experimental data 
2. Decide on a mixing method
3. Find the weights/mixing function to combine models

Taweret for Bayesian Model Mixing
---------------------------------

.. image:: _static/Taweret.png

Taweret is the protective ancient Egyptian goddess of childbirth and fertility. She has the head of a hippopotamus \
and limbs and paws of a lion. Her back and tail is of a Nile crocodile. Hence the name of our Bayesian Model \
Mixing package, Taweret!

Taweret follows a three step process for Bayesian Model Mixing, as discussed below.

Models
^^^^^^
The user has to provide models that they would like to mix. Currently Taweret supports mixing of two \
or more models with a single input parameter and a single output. The models should have a predict \
method and should return a mean and a variance for each input parameter value. 

Mixing Method
^^^^^^^^^^^^^
The user will then choose a mixing method. Currently Taweret supports: \
1. **Linear mixing** (2 models)
2. **Multivariate BMM** (2,...,N models)
3. **Regression Trees with BART** (2 models)

Details of each Mixing Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Linear mixing**: Mixing method employing a mixing function between the two models chosen \
(with four different types of mixing functions available in Taweret (*Step*, *Sigmoid*, \
*CDF*, *Piecewise cosine*). This mixing method takes two models and the experimental data as input \
and calculates the likelihood. Finding of the optimal weights by either optimizing the \
likelihood or finding the full posterior is done as the next step. 

**Multivariate BMM**: Mixing method that combines two (or more!) models provided by the user into \
a mixed model. This method, unlike linear mixing, only requires knowledge of the two models and their \
uncertainties at the input points. Given that the models are supplied by the user, one could combine \
two functions with a Gaussian Process (as seen in the example notebook for this method). 

**Regression Trees with BART**: (@John, add your description here)

Estimation of Mixing Function Weights 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using the likelihood provided by the mixed model, the user can find the optimal weights for Bayesian Model Mixing. \
We provide wrapper functions to the likelihood method so that one can use their favourite calibration software \
to estimate the weights. There is an example for this with SAMBA models in one of the example notebooks.

Performing Model Mixing and Calibration Together
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Rather than using calibrated models to mix we can do better by doing calibration and mixing together. The advantage of this \
is that the calibration of each model is not done by trying to fit the model to all experimental data and getting a global fit. \
Instead each model is calibrated only using the fraction of experimental data that can be well fitted with the model. 
This would avoid situations where a model is calibrated using experimental data that is outside its applicability.\
There is an example of this with Coleman models in one of the example notebooks.