#!/usr/bin/env python3

import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os, glob
import re
# Output data format
from configurations import *


#################################################################################
#### Try to figure out semi-automatically what observables to group together ####
#################################################################################

# This is the input:
# Specifies how observables are grouped according to these regular expression
regex_obs_to_group_list=[
(r'$\pi$/K/p dN/dy',"dN_dy_(pion|kaon|proton)"),
(r'$\pi$/K/p $\langle p_T \rangle$',"mean_pT_(pion|kaon|proton)"),
(r'$\Lambda/\Omega/\Xi$ dN/dy',"dN_dy_(Lambda|Omega|Xi)"),  
(r'$v_n\{2\}$',"v[2-5+]2") 
]

# This parts figures out how to group observables based on the regular expressions

obs_to_group={}
# Loop over observables to see which ones to group
for obs_name in obs_cent_list.keys():
    found_match=False
    for regex_id, (regex_label, regex_obs_to_group) in enumerate(regex_obs_to_group_list):
        r = re.compile(regex_obs_to_group)
        match=r.match(obs_name)
        # No match means nothing to group
        if (match is not None):
            if (found_match):
                print("Non-exclusive grouping. Can't work...")
                exit(1)
            else:
                found_match=True

                obs_to_group[obs_name]=(regex_id, regex_label)

    if (not found_match):
        obs_to_group[obs_name]=None

# Parse the previous list to make something useful out of it
final_obs_grouping = {}

#
for n, (key, value) in enumerate(obs_to_group.items()):

    if (value is None):
        newvalue=(n,key)
    else:
        newvalue=value

    final_obs_grouping.setdefault(newvalue, []).append(key)




##############
#### Plot ####
##############

def plot(all_calcs, idf=3):
    # Count how many observables to plot
    nb_obs=len(final_obs_grouping)
    # Decide how many columns we want the plot to have
    nb_of_cols=4
    # COunt how many rows needed
    nb_of_rows=int(np.ceil(nb_obs/nb_of_cols))
    # Prepare figure
    fig, axes = plt.subplots(figsize=(2*nb_of_cols, 2*nb_of_rows), 
                             ncols=nb_of_cols, nrows=nb_of_rows,
                             sharex=True)

    
    for n, ((regex_id, obs_name), obs_list) in \
                         enumerate(final_obs_grouping.items()):
        ax = axes.flatten()[n]
        if ax.is_last_row():
            ax.set_xlabel(r'Centrality (%)', fontsize=10)
        ax.set_title(obs_name, fontsize=10)
        print(all_calcs.shape)
        for ipt, calcs in enumerate(all_calcs):
            for obs, color in zip(active_obs_list['Pb-Pb-2760'],'rgbrgbrgb'):
                cent=obs_cent_list['Pb-Pb-2760'][obs]
                mid_centrality=[(low+up)/2. for low,up in cent]
                mean=calcs['Pb-Pb-2760'][obs]['mean'][idf,:]
                std=calcs['Pb-Pb-2760'][obs]['err'][idf,:]
                ax.errorbar(mid_centrality, mean, yerr=std, fmt='.-', color=color, alpha=0.15)
        ax.set_ylim(ymin=0)
    axes[1,2].set_ylim(ymax=.15)
    axes[1,0].set_ylim(ymax=2)
    plt.tight_layout(True)
    plt.savefig("plots/prior_df_{:d}.png".format(idf), dpi=400)
    plt.suptitle(r"Pb Pb 2760, $\delta-f = {:d}$".format(idf))
    #plt.show()

def corr(all_calcs, obs1, obs2, idf=0):
    for calcs in all_calcs:
        mean1=calcs['Pb-Pb-2760'][obs1]['mean'][idf,:]
        mean2=calcs['Pb-Pb-2760'][obs2]['mean'][idf,:]
        plt.scatter(mean1[:2], mean2[6:8]**.5, alpha=0.3, color='r')
    plt.xlabel(obs1)
    plt.ylabel(obs2)
    plt.show()




if __name__ == '__main__':
    filepath = sys.argv[1]
    # Load calculations       
    calcs = np.fromfile(filepath, dtype=np.dtype(bayes_dtype))
    #corr(calcs, 'pT_fluct','dNch_deta')
    plot(calcs)
