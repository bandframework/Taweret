#!/usr/bin/env python3
import logging
from configurations import *
import numpy as np
from bayes_mcmc import Chain
import pandas as pd
import matplotlib.pyplot as plt

# load validation points
system = system_strs[0]
design, design_min, design_max, labels = load_design(system, pset = 'validation')
system = system_strs[1]
design1, _, _, _ = load_design(system, pset = 'validation')
truth = [design.values[validation_pt,0]]+list(design1.values[validation_pt]) # true parameter  of a point
print(truth)
chain = Chain()
data = chain.load().T[:-1]
ndims, nsamples = data.shape
print(nsamples)

# Take samples from the chain
n_sample_from_chain=4000
sample_id_from_chain=np.random.choice(np.arange(nsamples), size=n_sample_from_chain, replace=False)
X=data[:,sample_id_from_chain]

with open("validate/{:d}-original.dat".format(validation_pt),'w') as f:
    # Open chain, and then
    # writes the credible limit of the original parameters
    f.write("# index True, Median, low-5%, low-20%, high-80%, high-95%\n")
    for i in range(ndims):
        samples = X[i]
        median = np.median(samples)
        l5 = np.quantile(samples, .05)
        l20 = np.quantile(samples, .2)
        h80 = np.quantile(samples, .8)
        h95 = np.quantile(samples, .95)
        f.write("{:d}\t{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\n".format(
                 i, truth[i], median, l5, l20, h80, h95
                 )
               )

with open("validate/{:d}-etas.dat".format(validation_pt),'w') as f:
    # transform design into eta/s(T_i) and zeta/s(T_i)
    # Ti is chose, e.g, to be
    #Ti = [.155, .175, .2, .25, .35]
    Ti = np.linspace(.13,.35, num=50)
    f.write("# T, Median, low-5%, low-20%, high-80%, high-95%\n")
    for T in Ti:
        samples = eta_over_s(T, X[8], X[9], X[10], X[11])
        median = np.median(samples)
        l5 = np.quantile(samples, .05)
        l20 = np.quantile(samples, .2)
        h80 = np.quantile(samples, .8)
        h95 = np.quantile(samples, .95)
        ty = eta_over_s(T, *truth[8:12])
        f.write("{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\n".format(
                 T, ty, median, l5, l20, h80, h95
                 )
               )

with open("validate/{:d}-zetas.dat".format(validation_pt),'w') as f:
    # transform design into eta/s(T_i) and zeta/s(T_i)
    # Ti is chose, e.g, to be
    #Ti = [.155, .175, .2, .25, .35]
    Ti = np.linspace(.13,.35, num=50)
    f.write("# T, Median, low-5%, low-20%, high-80%, high-95%\n")
    for T in Ti:
        samples = zeta_over_s(T, X[12], X[13], X[14], X[15])
        median = np.median(samples)
        l5 = np.quantile(samples, .05)
        l20 = np.quantile(samples, .2)
        h80 = np.quantile(samples, .8)
        h95 = np.quantile(samples, .95)
        ty = zeta_over_s(T, *truth[12:16])
        f.write("{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\n".format(
                 T, ty, median, l5, l20, h80, h95
                 )
               )
