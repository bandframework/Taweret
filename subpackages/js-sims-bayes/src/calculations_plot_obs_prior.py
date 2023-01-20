#!/usr/bin/env python3

import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os, glob
import re
import csv
import pandas as pd

# Output data format
#from calculations_file_format_event_average import *
from configurations import *
from bins_and_cuts import *

from pylab import rcParams

##############
#### Plot ####
##############
temp_obs_list = ['dN_dy_pion', 'dN_dy_kaon', 'dN_dy_proton',
                'mean_pT_pion', 'mean_pT_kaon', 'mean_pT_proton',
                'v22', 'v32'
                ]

exp_obs_list = temp_obs_list

#define a dictionary of math expressions for observables
obs_exp = {'dNch_deta': r'$dN_{ch}/d\eta$',
            'dET_deta': r'$dE_{T}/d\eta$',
            'pT_fluct': r'$p_T$ fluct'
            }

ft_size = 10

def plot(calcs, system):

    #Loop over delta-f
    for idf in [0, 1, 2, 3]:

        #print("idf = " + str(idf) )

        plt.figure(figsize=(8,8))
        linetype = '-'
        alpha = 0.3
        color_model = 'b'
        label_model = "Model"

        #for obs in obs_list:
        for n, obs in enumerate(temp_obs_list):

            plt.subplot(nb_of_rows,nb_of_cols,n+1)
            plt.xlabel(r'Centrality (%)', fontsize=ft_size)

            plt.ylabel(obs, fontsize=ft_size)
            plt.title(obs, fontsize=ft_size)

            cent = obs_cent_list[system][obs]
            mid_centrality = [(low+up)/2. for low,up in cent]
            #loop over design points
            for design_pt in range(0, n_design_pts_main): #exclude point 29 didnt finish
                mean_values=calcs[system][obs]['mean'][:,idf][design_pt]
                stat_uncert=calcs[system][obs]['err'][:,idf][design_pt]
                plt.errorbar(mid_centrality, mean_values, yerr=stat_uncert, ls=linetype, alpha=alpha, color=color_model, zorder=1)

            #load the experimental data
            if obs in exp_obs_list:

                #for Au-Au-200 the pi+, pi_ yields measured separately. Same for kaons, protons.
                #Here we take the positive particle yield and multiply by two to compare with model (mu_B = 0)
                if (obs in STAR_id_yields.keys() and system == 'Au-Au-200'):
                    if (obs == 'dN_dy_proton'):
                        expt_data = pd.read_csv('HIC_experimental_data/' + system + '/PHENIX/' + obs + '_+.dat',
                                                sep = ' ', skiprows=2, escapechar='#')
                    else :
                        expt_data = pd.read_csv('HIC_experimental_data/' + system + '/' + expt_for_system[system] + '/' + obs + '_+.dat',
                                                sep = ' ', skiprows=2, escapechar='#')

                    y_expt = expt_data['val'] * 2.0
                else :
                    expt_data = pd.read_csv('HIC_experimental_data/' + system + '/' + expt_for_system[system] + '/' + obs + '.dat',
                                            sep = ' ', skiprows=2, escapechar='#')
                    y_expt = expt_data['val']

                x_expt = expt_data['cent_mid']

                try :
                    err_expt = expt_data['err']
                except KeyError :
                    stat = expt_data['stat_err']
                    sys = expt_data['sys_err']
                    err_expt = np.sqrt(stat**2 + sys**2)

                #right now scaling error for pi^+ by factor of sqrt(2) when summing pi^+ and pi^-
                if (obs in STAR_id_yields.keys() and system == 'Au-Au-200'):
                    err_expt *= np.sqrt(2.0)
                #plt.scatter(x_expt, y_expt, color=color, edgecolors='black', zorder=2)
                color_expt = 'black'
                label_expt = 'STAR'
                if (obs == 'dN_dy_proton'):
                    color_expt = 'red'
                    label_expt = 'PHENIX'
                plt.errorbar(x_expt, y_expt, yerr=err_expt, zorder=2, color=color_expt, fmt='o', label=label_expt)

            #plt.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(system + ' Prior : ' + idf_label[idf] + ' Visc. Correction')
        plt.savefig('plots/' + system + "_obs_idf_" + str(idf) + ".pdf")

if __name__ == '__main__':
        results = []
        for file in glob.glob(sys.argv[1]):
                # Load calculations
                #calcs = np.fromfile(file, dtype=np.dtype(bayes_dtype))
                system = 'Au-Au-200'

                #how do we generalize this to plot the prior ?
                calcs = np.fromfile(file, dtype=bayes_dtype)

                print("calcs.shape = ")
                print(calcs.shape)

                # Count how many observables to plot
                #nb_obs=len(final_obs_grouping)
                nb_obs=len(temp_obs_list)
                # Decide how many columns we want the plot to have
                nb_of_cols=3
                # Count how many rows needed
                nb_of_rows=int(np.ceil(nb_obs/nb_of_cols))
                # Prepare figure
                fig = plt.figure(figsize=(2*nb_of_cols,2*nb_of_rows))

                entry = plot(calcs, system)
