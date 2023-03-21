#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 21:20:08 2022
@author: ozgesurer
"""
from matplotlib.transforms import Transform
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import os
import pandas as pd

# 8 bins
ALICE_cent_bins = np.array([[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70]])

obs_cent_list = {
'Pb-Pb-2760': {
    'dNch_deta' : ALICE_cent_bins,
    'dET_deta' : np.array([[0, 2.5], [2.5, 5], [5, 7.5], [7.5, 10],
                           [10, 12.5], [12.5, 15], [15, 17.5], [17.5, 20],
                           [20, 22.5], [22.5, 25], [25, 27.5], [27.5, 30],
                           [30, 32.5], [32.5, 35], [35, 37.5], [37.5, 40],
                           [40, 45], [45, 50], [50, 55], [55, 60],
                           [60, 65], [65, 70]]), # 22 bins
    'dN_dy_pion'   : ALICE_cent_bins,
    'dN_dy_kaon'   : ALICE_cent_bins,
    'dN_dy_proton' : ALICE_cent_bins,
    'dN_dy_Lambda' : np.array([[0,5],[5,10],[10,20],[20,40],[40,60]]), # 5 bins
    'dN_dy_Omega'  : np.array([[0,10],[10,20],[20,40],[40,60]]), # 4 bins
    'dN_dy_Xi'     : np.array([[0,10],[10,20],[20,40],[40,60]]), # 4 bins
    'mean_pT_pion'   : ALICE_cent_bins,
    'mean_pT_kaon'   : ALICE_cent_bins,
    'mean_pT_proton' : ALICE_cent_bins,
    'pT_fluct' : np.array([[0,5],[5,10],[10,15],[15,20], [20,25],[25,30],[30,35],[35,40], [40,45],[45,50],[50,55],[55,60]]), #12 bins
    'v22' : ALICE_cent_bins,
    'v32' : np.array([[0,5],[5,10],[10,20],[20,30], [30,40],[40,50]]), # 6 bins
    'v42' : np.array([[0,5],[5,10],[10,20],[20,30], [30,40],[40,50]]), # 6 bins
    }
}

obs_groups = {'yields' : ['dNch_deta', 'dET_deta', 'dN_dy_pion', 'dN_dy_kaon', 'dN_dy_proton'],
              'mean_pT' : ['mean_pT_pion', 'mean_pT_kaon','mean_pT_proton', ],
              'fluct' : ['pT_fluct'],
              'flows' : ['v22', 'v32', 'v42']}

obs_group_labels = {'yields' : r'$dN_\mathrm{id}/dy_p$, $dN_\mathrm{ch}/d\eta$, $dE_T/d\eta$ [GeV]',
                    'mean_pT' : r'$ \langle p_T \rangle_\mathrm{id}$' + ' [GeV]',
                    'fluct' : r'$\delta p_{T,\mathrm{ch}} / \langle p_T \rangle_\mathrm{ch}$',
                    'flows' : r'$v^{(\mathrm{ch})}_k\{2\} $'}

colors = ['b', 'g', 'r', 'c', 'm', 'tan', 'gray']

obs_tex_labels = {'dNch_deta' : r'$dN_\mathrm{ch}/d\eta$',
                  'dN_dy_pion' : r'$dN_{\pi}/dy_p$',
                  'dN_dy_kaon' : r'$dN_{K}/dy_p$',
                  'dN_dy_proton' : r'$dN_{p}/dy_p$',
                  'dET_deta' : r'$dE_{T}/d\eta$',

                  'mean_pT_proton' : r'$\langle p_T \rangle_p$',
                  'mean_pT_kaon' : r'$\langle p_T \rangle_K$',
                  'mean_pT_pion' : r'$\langle p_T \rangle_\pi$',

                  'pT_fluct' : r'$\delta p_{T,\mathrm{ch}} / \langle p_T \rangle_\mathrm{ch}$',
                  'v22' : r'$v^{(\mathrm{ch})}_2\{2\}$',
                  'v32' : r'$v^{(\mathrm{ch})}_3\{2\}$',
                  'v42' : r'$v^{(\mathrm{ch})}_4\{2\}$'}


#Model parameter names in Latex compatble form
model_param_dsgn = ['$\\theta$',#0
                    '$\\gamma$',#1
                    '$\\beta$',#2
    '$N$[$2.76$TeV]',#3
 '$p$',#4
 '$\\sigma_k$',#5
 '$w$ [fm]',#6
 '$d_{\\mathrm{min}}$ [fm]',#7
 '$\\tau_R$ [fm/$c$]',#8
 '$\\alpha$',#9
 '$T_{\\eta,\\mathrm{kink}}$ [GeV]',#10
 '$a_{\\eta,\\mathrm{low}}$ [GeV${}^{-1}$]',#11
 '$a_{\\eta,\\mathrm{high}}$ [GeV${}^{-1}$]',#12
 '$(\\eta/s)_{\\mathrm{kink}}$',#13
 '$(\\zeta/s)_{\\max}$',#14
 '$T_{\\zeta,c}$ [GeV]',#15
 '$w_{\\zeta}$ [GeV]',#16
 '$\\lambda_{\\zeta}$',#17
 '$b_{\\pi}$',#18
 '$T_{\\mathrm{sw}}$ [GeV]']#19


def plot_corner_viscosity(posterior_df,prior_df, method_name, n_samples=1000, prune=1, MAP=None, closure=None):

    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("ticks")
    #map_parameters = rslt.x
    sns.set_palette('bright')
    observables_to_plot=[0, 1, 2, 10, 11 , 12, 13, 14, 15, 16, 17, 18]
    posterior_df = posterior_df.copy(deep=True)
    prior_df = prior_df.copy(deep=True)
    posterior_df['distribution'] = 'posterior'
    prior_df['distribution'] = 'prior'
    df = pd.concat([prior_df, posterior_df], ignore_index=True)
    obs = observables_to_plot + [20]
    df = df.iloc[0:2*n_samples:prune,obs] 
    g = sns.PairGrid(df, corner=True, diag_sharey=False, hue='distribution', hue_kws={'alpha':0.5},
                    palette={'prior':sns.color_palette()[4],'posterior':sns.color_palette()[5]})
    g.map_lower(sns.kdeplot, fill=True, thresh=0.001, levels=50, bw_adjust=0.5)
    g.map_diag(sns.kdeplot, linewidth=2, shade=True , fill=True, bw_adjust=0.5)
    for n,i in enumerate(observables_to_plot):
        ax=g.axes[n][n]
        if MAP is not None:
            map_parameters=MAP.flatten()
            ax.axvline(x=map_parameters[i], ls='--', c=sns.color_palette()[9], label= 'MAP')
            ax.text(0,0.9,s= f'{map_parameters[i]:.3f}', transform=ax.transAxes)
        if closure is not None:
            map_parameters=closure.flatten()
            ax.axvline(x=map_parameters[i], ls='--', c=sns.color_palette()[3], label= 'Truth')
            ax.text(0,0.7,s= f'{map_parameters[i]:.3f}', transform=ax.transAxes)        
    #ax.axvline(x=truth[i], ls='--', c=sns.color_palette()[3], label = 'Truth')
    #ax.text(0,0.8,s= f'{truth[i]:.3f}', transform=ax.transAxes)
        if n==4:
            ax.legend(loc=0,fontsize='xx-small')    
    plt.tight_layout()
    plt.savefig(f'{method_name}/Viscosity.png', dpi=200)
    plt.show()
    return None

def plot_corner_no_viscosity(posterior_df,prior_df,  method_name, n_samples = 1000, prune=1, MAP=None, closure=None, transform=False):

    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("ticks")
    #map_parameters=map_values_saved.flatten()
    #n_samples_prior = 20000
    #prune = 1
    sns.set_palette('bright')
    if transform==False:
        observables_to_plot=[0, 1, 2 ,3 , 4, 5, 6, 7, 8, 9, 19]
        obs = observables_to_plot + [20]
    else:
        observables_to_plot=[0, 1, 2 ,3 , 4, 5, 6]
        obs = observables_to_plot + [27]
    posterior_df = posterior_df.copy(deep=True)
    prior_df = prior_df.copy(deep=True)
    posterior_df['distribution'] = 'posterior'
    prior_df['distribution'] = 'prior'
    df = pd.concat([prior_df, posterior_df], ignore_index=True)
    
    df = df.iloc[0:2*n_samples:prune,obs] 
    g = sns.PairGrid(df, corner=True, diag_sharey=False, hue='distribution', hue_kws={'alpha':0.5},
                    palette={'prior':sns.color_palette()[4],'posterior':sns.color_palette()[5]})
    g.map_lower(sns.kdeplot, fill=True, thresh=0.001, levels=50, bw_adjust=0.5)
    #g.map_upper(sns.kdeplot, shade=True, color=sns.color_palette()[0])
    g.map_diag(sns.kdeplot, linewidth=2, shade=True, bw_adjust=0.5)
    for n,i in enumerate(observables_to_plot):
        ax=g.axes[n][n]
        if MAP is not None:
            map_parameters = MAP.flatten()
            ax.axvline(x=map_parameters[i], ls='--', c=sns.color_palette()[9], label='MAP')
            ax.text(0.0,1,s= f'{map_parameters[i]:.3f}',fontdict={'color':sns.color_palette()[9]}, transform=ax.transAxes)
        if closure is not None:
            map_parameters=closure.flatten()
            ax.axvline(x=map_parameters[i], ls='--', c=sns.color_palette()[3], label= 'Truth')
            ax.text(0,0.7,s= f'{map_parameters[i]:.3f}', transform=ax.transAxes)    
    #ax.axvline(x=truth[i], ls='--', c=sns.color_palette()[3], label = 'Truth')
    #ax.text(0.6,1,s= f'{truth[i]:.3f}',fontdict={'color':sns.color_palette()[3]}, transform=ax.transAxes)
        if n==0:
            ax.legend(loc=1,fontsize='xx-small')
    plt.tight_layout()
    if transform==False:
        plt.savefig(f'{method_name}/WithoutViscosity.png', dpi=200)
    else:
        plt.savefig(f'{method_name}/WithoutViscosity_transform.png', dpi=200)
    plt.show()
    return None

def plot_corner_all(posterior_df, prior_df, method_name, n_samples = 1000, prune=1, MAP=None, closure=None, transform=False):
    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("ticks")
    #map_parameters=map_values_saved.flatten()
    #n_samples_prior = 20000
    #prune = 1
    #map_parameters = rslt.x
    sns.set_palette('bright')
    if transform==False:
        observables_to_plot=[i for i in range(0,20)]
        obs = observables_to_plot + [20]
    else:
        observables_to_plot=[i for i in range(0,27)]
        obs = observables_to_plot + [27]   
    
    posterior_df = posterior_df.copy(deep=True)
    prior_df = prior_df.copy(deep=True)
    posterior_df['distribution'] = 'posterior'
    prior_df['distribution'] = 'prior'
    df = pd.concat([prior_df, posterior_df], ignore_index=True)
    df = df.iloc[0:2*n_samples:prune,obs] 
    g = sns.PairGrid(df, corner=True, diag_sharey=False, hue='distribution', hue_kws={'alpha':0.5},
                    palette={'prior':sns.color_palette()[4],'posterior':sns.color_palette()[5]})
    g.map_lower(sns.kdeplot, fill=True)
    g.map_lower(sns.kdeplot, fill=True)
    #g.map_upper(sns.kdeplot, shade=True, color=sns.color_palette()[0])
    g.map_diag(sns.kdeplot, linewidth=2, shade=True, color=sns.color_palette()[4])
    for n,i in enumerate(observables_to_plot):
        ax=g.axes[n][n]
        if MAP is not None:
            map_parameters = MAP.flatten()
            ax.axvline(x=map_parameters[i], ls='--', c=sns.color_palette()[9], label='MAP')
            ax.text(0.0,1,s= f'{map_parameters[i]:.3f}',fontdict={'color':sns.color_palette()[9]}, transform=ax.transAxes)
        if closure is not None:
            map_parameters=closure.flatten()
            ax.axvline(x=map_parameters[i], ls='--', c=sns.color_palette()[3], label= 'Truth')
            ax.text(0,0.7,s= f'{map_parameters[i]:.3f}', transform=ax.transAxes)    
    #ax.axvline(x=truth[i], ls='--', c=sns.color_palette()[3], label = 'Truth')
    #ax.text(0.6,1,s= f'{truth[i]:.3f}',fontdict={'color':sns.color_palette()[3]}, transform=ax.transAxes)
        if n==0:
            ax.legend(loc=1,fontsize='xx-small')
    plt.tight_layout()
    if transform==False:
        plt.savefig(f'{method_name}/all.png', dpi=200)
    else:
        plt.savefig(f'{method_name}/all_transform.png', dpi=200)
    plt.show()
    return None

def zeta_over_s(T, zmax, T0, width, asym):
    DeltaT = T - T0
    sign = 1 if DeltaT>0 else -1
    x = DeltaT/(width*(1.+asym*sign))
    return zmax/(1.+x**2)
zeta_over_s = np.vectorize(zeta_over_s)

def eta_over_s(T, T_k, alow, ahigh, etas_k):
    if T < T_k:
        y = etas_k + alow*(T-T_k)
    else:
        y = etas_k + ahigh*(T-T_k)
    if y > 0:
        return y
    else:
        return 0.
eta_over_s = np.vectorize(eta_over_s)

def plot_shear(posterior_df, prior_df, method_name, n_samples = 1000, prune=1, MAP=None, closure=None, ax= None, legend=False):
    sns.set_context('paper', font_scale=2)
    Tt = np.linspace(0.15, 0.35, 100)
    if ax==None:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,6), sharex=False, sharey=False, constrained_layout=True)
        #fig.suptitle("Specific shear viscosity posterior", wrap=True)
        fig.suptitle("(a)", wrap=True)

    else:
        axes=ax




    prior_etas = []
    #design_min, design_max = prior[:,0], prior[:,1]
    for row in prior_df.iloc[0:n_samples:prune,[10,11,12,13]].values:
        [T_k, alow, ahigh, etas_k] = row
        prior=[]
        for T in Tt:
            prior.append(eta_over_s(T,T_k,alow,ahigh,etas_k))
        prior_etas.append(prior)
    per0_pr,per5_pr,per20_pr,per80_pr,per95_pr,per100_pr=np.percentile(prior_etas,[0,5,20,80,95,100], axis=0)

    posterior_etas = []
    
    for row in posterior_df.iloc[0:n_samples:prune,[10,11,12,13]].values:
        [T_k, alow, ahigh, etas_k] = row
        posterior=[]
        for T in Tt:
            posterior.append(eta_over_s(T,T_k,alow,ahigh,etas_k))
        posterior_etas.append(posterior)
    per0,per5,per20,per80,per95,per100=np.percentile(posterior_etas,[0,5,20,80,95,100], axis=0)
    axes.fill_between(Tt, per5_pr,per95_pr,color=sns.color_palette()[7], alpha=0.3, label='90% Prior')
    axes.fill_between(Tt,per5,per95,color=sns.color_palette()[9], alpha=0.2, label='90% C.I.')
    axes.fill_between(Tt,per20,per80, color=sns.color_palette()[9], alpha=0.3, label='60% C.I.')

    # Map, True temperature dependece of the viscosity
    if closure is not None:
        values = closure.flatten()
        print(values)
        [T_k, etas_k, alow, ahigh] = values[[6,7,8,9]]
        true_shear = eta_over_s(Tt, T_k, alow, ahigh, etas_k)
        axes.plot(Tt, true_shear, color = 'black', label = 'Truth', linewidth=2, linestyle='--')
    if MAP is not None:
        values = MAP.flatten()
        [T_k, alow, ahigh, etas_k] = values[[10,11,12,13]]
        true_shear = eta_over_s(Tt, T_k, alow, ahigh, etas_k)
        axes.plot(Tt, true_shear, color = 'g', label = 'MAP', linewidth=5)
    axes.legend(loc='upper left')
    axes.set_ylim(0,0.5)
    axes.set_xlabel('T [GeV]')
    axes.set_ylabel('$\eta/s$')
    
    plt.tight_layout()
    if method_name!=None:
        plt.savefig(f'{method_name}/shear.png', dpi=200)
        plt.show()
    return None

def plot_bulk(posterior_df,  prior_df, method_name, n_samples = 1000, prune=1, MAP=None, closure=None, ax=None, legend=False):
    sns.set_context('paper', font_scale=2)
    Tt = np.linspace(0.15, 0.35, 100)
    if ax==None:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,6),sharex=False, sharey=False, constrained_layout=True)
        #fig.suptitle("Specefic bulk viscosity posterior", wrap=True)
        fig.suptitle("(b)", wrap=True)
    else:
        axes=ax

    # True temperature dependece of the viscosity

    #[zmax, T0, width, asym] = truth[[11,12,13,14]]
    #true_bulk = zeta_over_s(Tt, zmax, T0, width, asym)


    prior_zetas = []
    #design_min, design_max = prior[:,0], prior[:,1]
    for row in prior_df.iloc[0:n_samples:prune,[14,15,16,17]].values:
        [zmax, T0, width, asym] = row   
        prior=[]
        for T in Tt:
            prior.append(zeta_over_s(T,zmax, T0, width, asym))
        prior_zetas.append(prior)
    per0_pr,per5_pr,per20_pr,per80_pr,per95_pr,per100_pr=np.percentile(prior_zetas,[0,5,20,80,95,100], axis=0)

    posterior_zetas = []
        
    for row in posterior_df.iloc[0:n_samples:prune,[14,15,16,17]].values:
        [zmax, T0, width, asym] = row   
        posterior=[]
        for T in Tt:
            posterior.append(zeta_over_s(T,zmax, T0, width, asym))
        posterior_zetas.append(posterior)
    per0,per5,per20,per80,per95,per100=np.percentile(posterior_zetas,[0,5,20,80,95,100], axis=0)
    axes.fill_between(Tt, per5_pr,per95_pr,color=sns.color_palette()[7], alpha=0.3, label='90% Prior')
    axes.fill_between(Tt,per5,per95,color=sns.color_palette()[4], alpha=0.2, label='90% C.I.')
    axes.fill_between(Tt,per20,per80, color=sns.color_palette()[4], alpha=0.3, label='60% C.I.')

    if closure is not None:
        values = closure.flatten()
        [zmax, T0, width, asym] = values[[14,15,16,17]]
        true_bulk = zeta_over_s(Tt, zmax, T0, width, asym)
        axes.plot(Tt, true_bulk,  color = 'black', label = 'Truth', linewidth=2, linestyle='--')
    if MAP is not None:
        values = MAP.flatten()
        [zmax, T0, width, asym] = values[[14,15,16,17]]
        true_bulk = zeta_over_s(Tt, zmax, T0, width, asym)
        axes.plot(Tt, true_bulk, color = 'g', label = 'MAP', linewidth=5)
    #axes.plot(Tt, true_bulk, color = 'r', label = 'Truth', linewidth=5)

    #pos=np.array(prior_zetas).T
    #axes.violinplot(pos[1::10,:].T, positions=Tt[1::10],widths=0.03)
    
    axes.legend(loc='upper right')
    axes.set_ylim(0,0.25)
    #else:
        #axes.legend(loc='upper right')
    axes.set_xlabel('T [GeV]')
    axes.set_ylabel('$\zeta/s$')
    plt.tight_layout()
    if method_name!=None:
        plt.savefig(f'{method_name}/bulk.png', dpi=200)
        plt.show()
    return None
