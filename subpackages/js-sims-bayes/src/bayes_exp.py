#!/usr/bin/env python3
#import logging
from configurations import *
import numpy as np
from calculations_load import validation_data

#get model calculations at VALIDATION POINTS
if validation:
    Y_exp_data = {s: validation_data[s][validation_pt] for s in system_strs}
#get experimental data
else :
    print("Loading experimental data from " + dir_obs_exp)
    #entry = np.zeros(1, dtype=np.dtype(bayes_dtype) )
    entry = np.zeros(1, dtype=np.dtype(bayes_calibration_dtype) )

    for s in system_strs:
        expt = expt_for_system[s]
        path_to_data = 'HIC_experimental_data/' + s + '/' + expt_for_system[s] + '/'
        path_to_PHENIX = 'HIC_experimental_data/' + s + '/PHENIX/'
        for obs in list( obs_cent_list[s].keys() ):

            n_bins_bayes = len(obs_cent_list[s][obs]) # only using these bins for calibration. Files may contain more bins
            for idf in range(number_of_models_per_run):

                #for yields measured for RHIC Au Au @ 200 GeV
                if (obs in STAR_id_yields.keys() and s == 'Au-Au-200'):
                    #for proton dN/dy use the PHENIX data rather than star,
                    #for all other observables use STAR
                    if (obs == 'dN_dy_proton'):
                        expt_data_pos = pd.read_csv(path_to_PHENIX + obs + '_+.dat', sep = ' ', skiprows=2, escapechar='#')
                        expt_data_neg = pd.read_csv(path_to_PHENIX + obs + '_-.dat', sep = ' ', skiprows=2, escapechar='#')
                    else :
                        expt_data_pos = pd.read_csv(path_to_data + obs + '_+.dat', sep = ' ', skiprows=2, escapechar='#')
                        expt_data_neg = pd.read_csv(path_to_data + obs + '_-.dat', sep = ' ', skiprows=2, escapechar='#')

                    #our model takes the sum of pi^+ and pi^-, k^+ and k^-, etc...
                    #the Au Au data are saved separately for particles and antiparticles
                    entry[s][obs]['mean'][:, idf] = expt_data_pos['val'].iloc[:n_bins_bayes] + expt_data_neg['val'].iloc[:n_bins_bayes]
                    entry[s][obs]['err'][:, idf] = np.sqrt( expt_data_pos['err'].iloc[:n_bins_bayes]**2 + expt_data_neg['err'].iloc[:n_bins_bayes]**2 )

                #for all other observables
                else :
                    try:
                        expt_data = pd.read_csv(path_to_data + obs + '.dat', sep = ' ', skiprows=2, escapechar='#')
                        entry[s][obs]['mean'][:, idf] = expt_data['val'].iloc[:n_bins_bayes]
                        try :
                            err_expt = expt_data['err'].iloc[:n_bins_bayes]
                        except KeyError :
                            stat = expt_data['stat_err'].iloc[:n_bins_bayes]
                            sys = expt_data['sys_err'].iloc[:n_bins_bayes]
                            err_expt = np.sqrt(stat**2 + sys**2)
                        entry[s][obs]['err'][:, idf] = err_expt
                    except FileNotFoundError:
                        print("no experimental data avalaible in " + path_to_data + " for " + obs)

                #set to zero experimental error by hand to examine uncertainty
                if set_exp_error_to_zero:
                    print("WARNING! Setting all experimental errors to zero!")
                    entry[s][obs]['err'][:, idf] = entry[s][obs]['err'][:, idf] * 0.

    Y_exp_data = entry[0]
