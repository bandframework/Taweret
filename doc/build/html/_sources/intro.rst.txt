Taweret for Bayesian Model Mixing
=================================

.. contents::
    :local:

Bayesian Model Mixing
---------------------

In the modern computer simulation studies we often come accross multiple theorotically \
sound models that can simulate the same phenomena. These models have different assumptions built into them \
and are only valid in certain domain of the input domain. It is most certainly possible that the true \
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

Taweret is the protective ancient Egyptian goddess of childbirth and fertility. She has a head of a hippopotamus \
and limbs and paws of a lion. Her back and tail is of a Nile crocodile. Hence the name of our Bayesian Model \
Mixing package, Taweret!

Taweret follows three step process for Bayesian Model Mixing

Models
^^^^^^
The user has to provide models that they would like to mix. Currently Taweret only support mixing of two \
models with a single input parameter and a single output. The models should have a predict method and \
should return a mean and a variance for each input parameter value. 

Mixing Method
^^^^^^^^^^^^^
The user will then choose a mixing method. Currently Taweret only support linear mixing and can handle \
two different types of mixing functions (*Step*, *Sigmoid*). Mixing method take the two models and \
the experimental data as input and calculates the likelihood. Finding of the optimal weights by either \
optimizing the likelihood or finindg the full posterior is done as the next step. 

Weight Estimation
^^^^^^^^^^^^^^^^^
Using the likelihood provided by the Mixed model user can find the optimal weights for Bayesian Model Mixing. \
We provide wrapper functions to the likelihood method so that one can use their favourite calibration software \
to estimate the weights. 

