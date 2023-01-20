#!/usr/bin/env python3
import logging
import dill
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from configurations import *
from emulator import *

def main():
    for s in systems:

        observables = []
        for obs, cent_list in obs_cent_list.items():
            observables.append(obs)

        #smaller subset of observables for plotting
        observables = [
        'dNch_deta',
        'dET_deta',
        'dN_dy_pion',
        'dN_dy_kaon',
        'dN_dy_proton',
        'dN_dy_Lambda',
        'dN_dy_Omega',
        'dN_dy_Xi',
        'mean_pT_pion',
        'mean_pT_kaon',
        'mean_pT_proton',
        'pT_fluct',
        'v22',
        'v32',
        'v42'
        ]

        #load the dill'ed emulator from emulator file
        system_str = "{:s}-{:s}-{:d}".format(*s)
        print("Loading emulators from emulator/emu-" + system_str + '.dill' )
        emu = dill.load(open('emulator/emu-' + system_str + '.dill', "rb"))
        print("NPC = " + str(emu.npc))
        print("idf = " + str(idf))


        #get VALIDATION points
        design, design_max, design_min, labels = \
               load_design(system=('Pb','Pb',2760), pset='validation')

        #get model calculations at VALIDATION POINTS
        print("Load calculations from " + f_obs_validation)
        model_data = np.fromfile(f_obs_validation, dtype=bayes_dtype)

        #make a plot
        fig, axes = plt.subplots(figsize=(10,6), ncols=5, nrows=3)
        print("Validating: ", design.shape)
        for obs, ax in zip(observables, axes.flatten()):
            Y_true = []
            Y_emu = []
            for ipt, pt in enumerate(design.values):
                if ipt not in delete_design_pts_set:
                    continue
                mean, cov = emu.predict(np.array([pt]), return_cov=True)
                y_true = model_data[system_str][obs]['mean'][ipt,idf]
                y_emu = mean[obs][0]
                #if 'dET' in obs or 'dN' in obs:
                #     y_emu = np.exp(y_emu) - 1.
                dy_emu = (np.diagonal(cov[obs, obs])**.5)[:,0]
                Y_true = np.concatenate([Y_true, y_true])
                Y_emu = np.concatenate([Y_emu, y_emu])
            ym, yM = np.min(Y_emu), np.max(Y_emu)
            ax.hist2d(Y_emu, Y_true, bins=31,
                      cmap='coolwarm', range=[(ym, yM),(ym, yM)])
            ym, yM = ym-(yM-ym)*.05, yM+(yM-ym)*.05
            ax.plot([ym,yM],[ym,yM],'k--', zorder=100)

            ax.annotate(obs, xy=(.05, .8), xycoords="axes fraction", color='White')
            if ax.is_last_row():
                ax.set_xlabel("Emulated")
            if ax.is_first_col():
                ax.set_ylabel("Computed")
            ax.ticklabel_format(scilimits=(2,1))

        plt.tight_layout(True)

        plt.show()

if __name__ == "__main__":
    main()
