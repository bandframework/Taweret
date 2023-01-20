#!/usr/bin/env python3
"""
Calculate the maximum log Likelihood of the model specified by idf in configurations.py,
(the log Likelihood at its MAP value) as well as the bootstrap average <log L>. 
"""

import time
import matplotlib.pyplot as plt
from configurations import *
from emulator import Trained_Emulators, Trained_Emulators_all_df, _Covariance
from bayes_exp import Y_exp_data
from bayes_mcmc import *

MAP_params_combined = {}

#NOTE that if the MAP lies at exactly the boundary value, the log_likelihood will evaluate to zero because of the prior bounds,
#in this case we need to add a small epsilon to put the value inside of the prior bounds
eps = 1.0e-5
#                                N1      N2    p   sigma_k   w     d3   tau_R  alpha  T_eta,kink a_low   a_high eta_kink zeta_max T_(zeta,peak) w_zeta lambda_zeta    b_pi   T_s
MAP_params_combined['Grad']  = [14.2,  5.73,  0.06,  1.05,  1.12,  3.00,  1.46,  0.031,  0.223,  -0.78,   0.37,    0.096,   0.13,      0.12+eps,      0.072,    -0.12,   4.65 , 0.136]
MAP_params_combined['C.E.']  = [15.6,  6.24,  0.06,  1.00,  1.19,  2.60,  1.04,  0.024,  0.268,  -0.73,   0.38,    0.042,   0.127,     0.12+eps,      0.025+eps,    0.095,   5.6,  0.146]
MAP_params_combined['P.B.']  = [13.2,  5.31,  0.14,  0.98,  0.81,  3.11,  1.46,  0.017,  0.194,  -0.47,   1.62,    0.105,   0.165,     0.194,      0.026,   -0.072,  5.54,  0.147]

def calc_max_log_likelihood():
    #get the emulator predictions at the MAP values
    chain = Chain(path=workdir/'mcmc'/'chain-idf-{:}_LHC_RHIC_PTEMCEE.hdf'.format(idf))
    #load MAP params, and append the 'extra std. deviation discrepancy parameter'
    print("*********************************************************")
    print("Calculating Log Likelihood at max. (at MAP values) for...")
    print("idf = " + str(idf))
    print(idf_label[idf])
    X = np.array( MAP_params_combined[ idf_label_short[idf] ] + [1.1e-3]   )
    log_l = chain.log_likelihood(X)
    print("log_likelihood = " + str(log_l))
    print("*********************************************************")

def calc_avg_log_likelihood():
    #get the emulator predictions at the MAP values
    chain = Chain(path=workdir/'mcmc'/'chain-idf-{:}_LHC_RHIC_PTEMCEE.hdf'.format(idf))
    #load MAP params, and append the 'extra std. deviation discrepancy parameter'
    print("*********************************************************")
    print("Calculating Avg. Log Likelihood for...")
    print("idf = " + str(idf))
    print(idf_label[idf])
    #X = np.array( MAP_params_combined[ idf_label_short[idf] ] + [1.1e-3]   )
    data = chain.load()
    thin = 10000

    avg_log_l_vals = []
    for iter in range(10):
        np.random.shuffle(data)
        X = data[::thin, :]
        start = time.time()
        log_l = chain.log_likelihood(X)
        avg_log_l = log_l.mean()
        avg_log_l_vals.append(avg_log_l)
        end = time.time()
        print("< log_L = " + str(avg_log_l) + " >")
        print("calculated log_likelihood in " + str(end-start) + " seconds")

    avg_log_l_vals = np.array(avg_log_l_vals)
    bootstrap_mean = avg_log_l_vals.mean()
    bootstrap_std = avg_log_l_vals.std()
    print("Bootstrap < log_L > = " + str( round(bootstrap_mean, 5) ) + " +/- " + str( round(bootstrap_std, 5) ) )
    print("*********************************************************")

def main():

    calc_max_log_likelihood()

    calc_avg_log_likelihood()

if __name__ == '__main__':
    main()
