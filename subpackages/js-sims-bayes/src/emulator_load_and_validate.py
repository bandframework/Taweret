#!/usr/bin/env python3
import logging
import dill
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from string import ascii_letters

from configurations import *
from emulator import *
from calculations_load import validation_data, trimmed_model_data
from bayes_mcmc import Chain, credible_interval
from bayes_plot import obs_tex_labels_2
def plot_residuals(system_str, emu, design, cent_bin, observables, nrows, ncols):
    """
    Plot a histogram of the percent difference between the emulator
    prediction and the model at design points in either training or validation sets.
    """

    print("Plotting emulator residuals")

    fig, axes = plt.subplots(figsize=(10,10), ncols=ncols, nrows=nrows)
    for obs, ax in zip(observables, axes.flatten()):
        Y_true = []
        Y_emu = []

        if crossvalidation:
            for pt in cross_validation_pts:
                params = design.iloc[pt].values
                mean, cov = emu.predict(np.array([params]), return_cov = True)
                y_true = validation_data[system_str][pt, idf][obs]['mean'][cent_bin]
                y_emu = mean[obs][0][cent_bin]
                is_mult = ('dN' in obs) or ('dET' in obs)
                if is_mult and transform_multiplicities:
                    y_emu = np.exp(y_emu) - 1.
                    y_true = np.exp(y_true) - 1.
                #dy_emu = (np.diagonal(cov[obs, obs])**.5)[:,0]
                Y_true.append(y_true)
                Y_emu.append(y_emu)

        else :
            for pt, params in enumerate(design.values):
                mean, cov = emu.predict(np.array([params]), return_cov = True)
                y_true = validation_data[system_str][pt, idf][obs]['mean'][cent_bin]
                y_emu = mean[obs][0][cent_bin]
                is_mult = ('dN' in obs) or ('dET' in obs)
                if is_mult and transform_multiplicities:
                    y_emu = np.exp(y_emu) - 1.
                    y_true = np.exp(y_true) - 1.
                #dy_emu = (np.diagonal(cov[obs, obs])**.5)[:,0]
                Y_true.append(y_true)
                Y_emu.append(y_emu)

        Y_true = np.array(Y_true)
        Y_emu = np.array(Y_emu)

        residuals = (Y_emu - Y_true) / Y_emu
        std_resid = np.sqrt( np.var(residuals) )
        bins = np.linspace(-0.5, 0.5, 31)
        ax.hist(residuals, bins = bins, density = True)
        ax.set_xlim(-0.5, 0.5)

        ax.set_title(obs)
        ax.annotate(" std : " + str( round(std_resid, 2) ), xy=(.05, .8), xycoords = "axes fraction")

    plt.tight_layout(True)
    plt.savefig('validation_plots/emulator_residuals.png', dpi=300)

    #plt.show()

def plot_residuals_corr(system_str, emu, design, cent_bin, observables):
    """
    Plot a histogram of the percent difference between the emulator
    prediction and the model at design points in either training or validation sets.
    """

    print("Plotting emulator residuals obs1 vs obs2 ")
    ncols = nrows = len(observables)

    fig, axes = plt.subplots(figsize=(30,30), ncols=ncols, nrows=nrows)
    bins = np.linspace(-0.5, 0.5, 31)
    for row, obs1 in enumerate(observables):
        for col, obs2 in enumerate(observables):
            residuals_1 = []
            residuals_2 = []
            for pt, params in enumerate(design.values):
                mean, cov = emu.predict(np.array([params]), return_cov = True)
                y_true1 = validation_data[system_str][pt, idf][obs1]['mean'][cent_bin]
                y_emu1 = mean[obs1][0][cent_bin]
                res_1 = (y_emu1 - y_true1) / y_emu1
                residuals_1.append(res_1)
                y_true2 = validation_data[system_str][pt, idf][obs2]['mean'][cent_bin]
                y_emu2 = mean[obs2][0][cent_bin]
                res_2 = (y_emu2 - y_true2) / y_emu2
                residuals_2.append(res_2)

            axes[row,col].scatter(residuals_1, residuals_2)
            axes[row,col].set_xlabel(obs1)
            axes[row,col].set_ylabel(obs2)
            #axes[row, col].set_title(obs2 + " vs " + obs1 + " residuals")
            #if axes[row,col].is_last_row():
            #    axes[row,col].set_xlabel("res : " + obs1)
            #if axes[row,col].is_first_col():
            #    axes[row,col].set_ylabel("res : " + obs2)
            #axes[row, col].set_xlim(-0.5, 0.5)
            #axes[row, col].hist2d(res_1, res_2, bins = [bins, bins], density = True)

    plt.savefig('validation_plots/emulator_residuals_corr.png', dpi=300)

def plot_scatter(system_str, emu, design, cent_bin, observables):
    """
    Plot a scatter plot of the emulator prediction vs the model prediction at
    design points in either training or testing set.
    """

    print("Plotting scatter plot of emulator vs model")
    ncols = 3
    nrows = 2

    if len(observables) > 6:
        ncols = 4
        nrows = 3

    fig, axes = plt.subplots(figsize=(3*ncols,3*nrows), ncols=ncols, nrows=nrows)
    for obs, ax in zip(observables, axes.flatten()):
        Y_true = []
        Y_emu = []

        if crossvalidation:
            for pt in cross_validation_pts:
                params = design.iloc[pt].values
                mean, cov = emu.predict(np.array([params]), return_cov = True)
                y_true = validation_data[system_str][pt, idf][obs]['mean'][cent_bin]
                y_emu = mean[obs][0][cent_bin]
                is_mult = ('dN' in obs) or ('dET' in obs)
                if is_mult and transform_multiplicities:
                    y_emu = np.exp(y_emu) - 1.
                    y_true = np.exp(y_true) - 1.
                #dy_emu = (np.diagonal(cov[obs, obs])**.5)[:,0]
                Y_true.append(y_true)
                Y_emu.append(y_emu)

        else :
            for pt, params in enumerate(design.values):
                mean, cov = emu.predict(np.array([params]), return_cov = True)
                y_true = validation_data[system_str][pt, idf][obs]['mean'][cent_bin]
                y_emu = mean[obs][0][cent_bin]
                is_mult = ('dN' in obs) or ('dET' in obs)
                if is_mult and transform_multiplicities:
                    y_emu = np.exp(y_emu) - 1.
                    y_true = np.exp(y_true) - 1.
                #dy_emu = (np.diagonal(cov[obs, obs])**.5)[:,0]
                Y_true.append(y_true)
                Y_emu.append(y_emu)
                if pt in delete_design_pts_validation_set:
                    ax.scatter(y_true, y_emu, color='red')

        Y_true = np.array(Y_true)
        Y_emu = np.array(Y_emu)
        ym, yM = np.min(Y_emu), np.max(Y_emu)
        #h = ax.hist2d(Y_emu, Y_true, bins=31, cmap='coolwarm', range=[(ym, yM),(ym, yM)])
        ax.scatter(Y_emu, Y_true)
        ym, yM = ym-(yM-ym)*.05, yM+(yM-ym)*.05
        ax.plot([ym,yM],[ym,yM],'k--', zorder=100)

        ax.annotate(obs_tex_labels_2[obs], xy=(.05, .8), xycoords="axes fraction", fontsize=12)
        if ax.is_last_row():
            ax.set_xlabel("Emulated")
        if ax.is_first_col():
            ax.set_ylabel("Computed")
        ax.ticklabel_format(scilimits=(2,1))

    plt.tight_layout(True)
    plt.savefig('validation_plots/emulator_vs_model_' + system_str + '_' + idf_label_short[idf] + '.png', dpi=300)


def plot_model_stat_uncertainty(system_str, design, cent_bin, observables, nrows, ncols):
    """
    Plot the model uncertainty for all observables
    """
    print("Plotting model stat. uncertainty")

    fig, axes = plt.subplots(figsize=(10,8), ncols=ncols, nrows=nrows)
    for obs, ax in zip(observables, axes.flatten()):


        values = []
        errors = []
        rel_errors = []

        #note if transformation of multiplicities is turned on!!!
        for pt in range( n_design_pts_main - len(delete_design_pts_set) ):
            val = trimmed_model_data[system_str][pt, idf][obs]['mean'][cent_bin]
            err = trimmed_model_data[system_str][pt, idf][obs]['err'][cent_bin]
            values.append(val)
            errors.append(err)
            if val > 0.0:
                if np.isnan(err / val):
                    print("nan")
                else :
                    rel_errors.append(err / val)

        rel_errors = np.array(rel_errors)
        std = np.sqrt( np.var(rel_errors) )
        mean = np.mean(rel_errors)

        rel_errors = rel_errors[ rel_errors < (mean + 5.*std) ]

        ax.hist(rel_errors, 20)
        ax.set_title(obs)

        if ax.is_last_row():
            ax.set_xlabel("relative error")

    #plt.suptitle('Distribution of model statistical error')
    plt.tight_layout(True)
    plt.savefig('validation_plots/model_stat_errors.png', dpi=300)

def closure_test_credibility_intervals(system_str, design):
    chain = Chain()

    #get VALIDATION points
    keys = chain.labels
    allowed = set(ascii_letters)

    data = chain.load().T
    truths = list(design.values[validation_pt])+[0.0]
    for x, xkey, truth in zip(data, keys, truths):
        #write the truths and credibility intervals to file
        new_key = ''.join(l for l in xkey if l in allowed)
        new_key = new_key.replace('GeV', '')
        new_key = new_key.replace('TeV', '')
        new_key = new_key.replace('fm', '')
        new_key = new_key.replace('mathrm', '')
        cred = np.percentile(x, [10, 30, 50, 70, 90], axis=0)
        cred_string = ''
        for val in cred:
            cred_string += (str(val) + "\t")
        with open("closure_truth_dob/" + new_key + ".dat", "a+") as myfile:
            myfile.write( str(truth) + "\t" + cred_string )

def closure_test_credibility_eta_zeta(system_str, design):
    chain = Chain()

    #get VALIDATION points
    keys = chain.labels

    data = chain.load().T[:-1]
    ndims, nsamples = data.shape
    truths = list(design.values[validation_pt])
    Ti = np.linspace(0.13, 0.4, num=50)

    with open("validate_eta_zeta/{:d}-etas.dat".format(validation_pt),'w') as f:
        # transform design into eta/s(T_i) and zeta/s(T_i)
        f.write("#T\ttruth\tmedian\tlow5\tlow20\thigh80\thigh95\n")
        for T in Ti:
            samples = eta_over_s(T, data[7,:], data[8,:], data[9,:], data[10,:])
            true_eta_over_s = eta_over_s(T, truths[7], truths[8], truths[9], truths[10])
            median = np.median(samples)
            l5 = np.quantile(samples, .05)
            l20 = np.quantile(samples, .2)
            h80 = np.quantile(samples, .8)
            h95 = np.quantile(samples, .95)
            f.write("{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\n".format(
                     T, true_eta_over_s, median, l5, l20, h80, h95
                     )
                   )

    with open("validate_eta_zeta/{:d}-zetas.dat".format(validation_pt),'w') as f:
        # transform design into eta/s(T_i) and zeta/s(T_i)
        # Ti is chose, e.g, to be
        f.write("#T\ttruth\tmedian\tlow5\tlow20\thigh80\thigh95\n")
        for T in Ti:
            samples = zeta_over_s(T, data[11, :], data[12, :], data[13, :], data[14, :])
            true_zeta_over_s = zeta_over_s(T, truths[11], truths[12], truths[13], truths[14])
            median = np.median(samples)
            l5 = np.quantile(samples, .05)
            l20 = np.quantile(samples, .2)
            h80 = np.quantile(samples, .8)
            h95 = np.quantile(samples, .95)

            f.write("{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\n".format(
                     T, true_zeta_over_s, median, l5, l20, h80, h95
                     )
                   )

def main():

    cent_bin = 3

    for s in system_strs:
        observables = []
        for obs, cent_list in obs_cent_list[s].items():
            if obs in active_obs_list[s]:
                observables.append(obs)

        nrows = 4
        ncols = 4

        if pseudovalidation:
            #using training points as testing points
            design, design_max, design_min, labels = load_design(s, pset='main')
        else :
            design, design_max, design_min, labels = load_design(s, pset='validation')

        print("Validation design set shape : (Npoints, Nparams) =  ", design.shape)

        #load the dill'ed emulator from emulator file
        print("Loading emulators from emulator/emulator-" + s + '-idf-' + str(idf) + '.dill' )
        emu = dill.load(open('emulator/emulator-' + s + '-idf-' + str(idf) + '.dill', "rb"))
        print("NPC = " + str(emu.npc))
        print("idf = " + str(idf))

        #make a plot of the residuals ; percent difference between emulator and model
        #plot_residuals(system_str, emu, design, cent_bin, observables, nrows, ncols)

        #make a scatter plot to check if residuals between different observables are correlated
        #plot_residuals_corr(system_str, emu, design, cent_bin, observables)

        #make a scatter plot of emulator prediction vs model prediction
        plot_scatter(s, emu, design, cent_bin, observables)

        #make a histogram to check the model statistical uncertainty
        #plot_model_stat_uncertainty(system_str, design, cent_bin, observables, nrows, ncols)

        #check if truth falls within credibility intervals
        #closure_test_credibility_intervals(system_str, design)

        #check if eta/s , zeta/s at specified temperatures fall within dob intervals
        #closure_test_credibility_eta_zeta(system_str, design)


if __name__ == "__main__":
    main()
