#!/usr/bin/env python3

import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os, glob
import re
# Output data format
from configurations import *


design_pt_to_plot=2

#################################################################################
#### Try to figure out semi-automatically what observables to group together ####
#################################################################################

# This is the input:
# Specifies how observables are grouped according to these regular expression
# Also specify if they should be plotted on a linear or a log scale
regex_obs_to_group_list=[
(r'$\pi$/K/p dN/dy',"dN_dy_(pion|kaon|proton)",'log'),
(r'$\pi$/K/p $\langle p_T \rangle$',"mean_pT_(pion|kaon|proton)",'linear'),
(r'$\Lambda/\Omega/\Xi$ dN/dy',"dN_dy_(Lambda|Omega|Xi)",'log'),  
(r'$v_n\{2\}$',"v[2-5+]2",'linear'),
(r'$dN_{ch}/d\eta$',"dNch_deta",'log'),
(r'$dE_T/d\eta$',"dET_deta",'log'),
(r'$\langle p_T \rangle$ fluct',"pT_fluct",'linear'),
]

# This parts figures out how to group observables based on the regular expressions

obs_to_group={}

# Loop over observables to see which ones to group
for system in system_strs:
    obs_to_group[system]={}
    for obs_name in obs_cent_list[system]:
        found_match=False
        for regex_id, (regex_label, regex_obs_to_group, plot_scale) in enumerate(regex_obs_to_group_list):
            r = re.compile(regex_obs_to_group)
            match=r.match(obs_name)
            # No match means nothing to group
            if (match is not None):
                if (found_match):
                    print("Non-exclusive grouping. Can't work...")
                    exit(1)
                else:
                    found_match=True

                    obs_to_group[system][obs_name]=(regex_id, regex_label, plot_scale)

        if (not found_match):
            obs_to_group[system][obs_name]=None

# Parse the previous list to make something useful out of it
final_obs_grouping = {}

#
for system in system_strs:

    final_obs_grouping[system]={}

    for n, (key, value) in enumerate(obs_to_group[system].items()):

        if (value is None):
            newvalue=(n,key)
        else:
            newvalue=value

        final_obs_grouping[system].setdefault(newvalue, []).append(key)


##############
#### Plot ####
##############

def plot(calcs):

    for system in system_strs:

        # Count how many observables to plot
        nb_obs=len(final_obs_grouping[system])
        # Decide how many columns we want the plot to have
        nb_of_cols=4
        # COunt how many rows needed
        nb_of_rows=int(np.ceil(nb_obs/nb_of_cols))
        # Prepare figure
        fig = plt.figure(figsize=(2*nb_of_cols,2*nb_of_rows))

        line_list=[]

        #Loop over grouped observables
        #for n, (obs, cent) in enumerate(obs_cent_list.items()):
        for n, ((regex_id, obs_name, plot_scale), obs_list) in enumerate(final_obs_grouping[system].items()):

            plt.subplot(nb_of_rows,nb_of_cols,n+1)
            plt.xlabel(r'Centrality (%)', fontsize=10)
            plt.ylabel(obs_name, fontsize=10)
            plt.yscale(plot_scale)

            # Loop over observable group
            for obs, color in zip(obs_list,'rgbrgbrgb'):

                cent=obs_cent_list[system][obs]
                mid_centrality=[(low+up)/2. for low,up in cent]

                #Loop over delta-f
                idf_list=[0,1,2,3]
                idf_sym=['D','o','^','.']
                for idf, line in zip(idf_list, idf_sym):

                    mean_values=calcs[system][obs]['mean'][:,idf][design_pt_to_plot]
                    stat_uncert=calcs[system][obs]['err'][:,idf][design_pt_to_plot]
                    line_type,_,_ = plt.errorbar(mid_centrality, mean_values, yerr=stat_uncert, fmt=line, color=color, markersize=4)
                    line_list.append(line_type)

            if (plot_scale != "log"):
                plt.ylim(ymin=0)

            # Plot legend in first subplot only
            if (0 == n):
                plt.legend(line_list,["idf="+str(idf) for idf in idf_list],loc="upper right",fontsize=10)
                



        plt.tight_layout(True)
        #plt.savefig("obs.pdf")
        plt.show()



if __name__ == '__main__':
        results = []
        for file in glob.glob(sys.argv[1]):
                # Load calculations       
                calcs = np.fromfile(file, dtype=np.dtype(bayes_dtype))

                entry = plot(calcs)
