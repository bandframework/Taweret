import sys
import os
from pathlib import Path
os.environ["WORKDIR"] = "/users/PAS0254/dananjaya/Taweret/subpackages/js-sims-bayes"
# You will have to change the following imports depending on where you have 
# the packages installed
# If using binder please uncomment the followings.
#sys.path.append("/home/jovyan/")
workdir = Path(os.getenv('WORKDIR', '.'))
sys.path.append(str(workdir)+"/src")
sys.path.append("/users/PAS0254/dananjaya/Taweret/")
sys.path.append("/users/PAS0254/dananjaya/.conda/envs/jstaw/")
from configurations import *
from emulator import *
#sys.path.append("/Users/dananjayaliyanage/git/Taweret")
#sys.path.append("/Users/dananjayaliyanage/git/Taweret/subpackages/SAMBA")

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('poster')
# To define priors. (uncoment if not using default priors)
#import bilby

# For other operations
import numpy as np
import bilby

obs_tex_labels = {
                    'dNch_deta' : r'$dN_{ch}/d\eta$',
                    'dN_dy_pion' : r'$dN_{\pi}/dy$',
                    'dN_dy_kaon' : r'$dN_{k}/dy$',
                    'dN_dy_proton' : r'$dN_{p}/dy$',
                    'dN_dy_Lambda' : r'$dN_{\Lambda}/dy$',
                    'dN_dy_Omega' : r'$dN_{\Omega}/dy$',
                    'dN_dy_Xi' : r'$dN_{\Xi}/dy$',
                    'dET_deta' : r'$dE_{T}/d\eta$',
                    'mean_pT_pion' : r'$\langle p_T \rangle _{\pi}$',
                    'mean_pT_kaon' : r'$\langle p_T \rangle _{k}$',
                    'mean_pT_proton' : r'$\langle p_T \rangle _{p}$',
                    'pT_fluct' : r'$\delta p_T / \langle p_T \rangle$',
                    'v22' : r'$v_2\{2\}$',
                    'v32' : r'$v_3\{2\}$',
                    'v42' : r'$v_4\{2\}$',
}

# Toy models from SAMBA
from Taweret.models import jetscape_sims_models as sims
from Taweret.core.base_model import BaseModel

# Mixing method
from Taweret.mix.bivariate_linear import BivariateLinear as BL

def BMM_for_RHIC(m1_num : int, m2_num: int, outdir: str, label: str, 
            obs : list, g=np.linspace(0, 60, 20), 
            plot_g = np.linspace(0.0,60,100)):
    print(f'Observables that we consider are {obs}')
    names = {0:'Grad', 1:'CE', 3:'PTB'}
    m1_name = names[m1_num]
    m2_name = names[m2_num]

    all_obs_names = list(obs_cent_list['Pb-Pb-2760'].keys())
    #fix_obs_to_remove = ['dN_dy_Lambda', 'dN_dy_Omega', 'dN_dy_Xi']
    obs_to_remove = [ob for ob in all_obs_names if ob not in obs] 
    print(obs_to_remove)
    m1 = sims.jetscape_models_pb_pb_2760(fix_MAP=True,model_num=m1_num, obs_to_remove=obs_to_remove)
    m2 = sims.jetscape_models_pb_pb_2760(fix_MAP=True,model_num=m2_num, obs_to_remove=obs_to_remove)
    exp = sims.exp_data()

    #g = np.linspace(0, 60, 20)
    #plot_g = np.linspace(0.0,60,100)
    m1_prediction = m1.evaluate(plot_g)
    m2_prediction = m2.evaluate(plot_g)
    #true_output = truth.evaluate(plot_g)
    exp_data= exp.evaluate(g,obs_to_remove=obs_to_remove)
    n_obs = len(obs)
    for i in range(0,n_obs):
        fig, ax_f = plt.subplots(figsize=(10,10))
        # ax_f.errorbar(plot_g, m1_prediction[0][:,i].flatten(), 
        #             yerr=m1_prediction[1][:,i].flatten(), 
        #             label='Grad', alpha=0.2)
        # ax_f.errorbar(plot_g, m2_prediction[0][:,i].flatten(), 
        #             yerr=m1_prediction[1][:,i].flatten(),
        #             label='CE', alpha=0.2)
        ax_f.plot(plot_g, m1_prediction[0][:,i].flatten(), label=m1_name)
        ax_f.plot(plot_g, m2_prediction[0][:,i].flatten(), label=m2_name)
    
        #ax_f.plot(plot_g, true_output[0], label='truth')
        ax_f.errorbar(g,exp_data[0][:,i].flatten(), 
                    yerr=exp_data[1][:,i].flatten(),
                    marker='x', label='experimental data')
        ax_f.set_xlabel('Centrality')
        #ax_f.set_ylim(1.2,3.2)
        ax_f.set_ylabel(obs_tex_labels[obs[i]])
        ax_f.legend()

        os.makedirs(outdir+'/figures/', exist_ok = True)
        plt.tight_layout()
        fig.savefig(outdir+'/figures/'+label, dpi=100)
    
    models= {m1_name:m1,m2_name:m2}
    mix_model = BL(models_dic=models, method='addstep')

    #uncoment to change the prior from the default
    print('Setting priors')
    #priors = bilby.core.prior.PriorDict()
    #priors['step_0'] = bilby.core.prior.Uniform(-60, 60, name="step_0")
    #mix_model.set_prior(priors)
    priors = bilby.core.prior.PriorDict()
    priors['addstep_0'] = bilby.core.prior.Uniform(0, 60, name="addstep_0")
    priors['addstep_1'] = bilby.core.prior.Uniform(0, 1, name="addstep_1")
    mix_model.set_prior(priors)
    
    print('Running MCMC')

    result = mix_model.train(x_exp=g, y_exp=exp_data[0], y_err=exp_data[1], outdir = outdir, label=label, 
                        load_previous=True)

    result.plot_corner()

    print('Calculate from posterior')
    
    _,mean_prior,CI_prior, _ = mix_model.prior_predict(plot_g, CI=[5,20,80,95])
    _,mean,CI, _ = mix_model.predict(plot_g, CI=[5,20,80,95])
    
    per5, per20, per80, per95 = CI
    prior5, prior20, prior80, prior95 = CI_prior

    print(f'MAP value is {mix_model.map}')
    # Map value prediction for the step mixing function parameter
    map_prediction = mix_model.evaluate(mix_model.map, plot_g)
    
    for i in range(0,n_obs):
        os.makedirs(outdir+'/figures/', exist_ok = True)
        fig, axs = plt.subplots(1,1,figsize=(10,10))
        ax_f = axs
        ax_f.errorbar(plot_g, m1_prediction[0][:,i].flatten(), 
                    yerr=m1_prediction[1][:,i].flatten(), 
                    label=m1_name, alpha=0.5)
        ax_f.errorbar(plot_g, m2_prediction[0][:,i].flatten(), 
                    yerr=m2_prediction[1][:,i].flatten(),
                    label=m2_name, alpha=0.5)
        ax_f.plot(plot_g, mean[0][i,:].flatten(), label='Mean BMM')
        ax_f.errorbar(plot_g, map_prediction[i,:].flatten(), label='MAP', color='k')
        #ax_f.plot(plot_g, true_output[0], label='truth')
        ax_f.scatter(g,exp_data[0][:,i].flatten(), marker='x', label='experimental data', color='r')
        ax_f.set_xlabel('Centrality')
        ax_f.set_ylabel(obs_tex_labels[obs[i]])
        ax_f.legend()
        plt.tight_layout()
        fig.savefig(outdir+'/figures/'+'MAP_'+label, dpi=100)

        #ax_hist = axs.flatten()[1]
        #sns.histplot(data=result.posterior, x='step_0', kde=True, ax=ax_hist)
        #ax_hist.axvline(x = mix_model.map, color = 'r', label = f'MAP : {mix_model.map[0]}') 
        #ax_hist.legend()
        map_parameters=mix_model.map.flatten()
        sns.set_palette('bright')
        observables_to_plot=[0, 1]
        gg = sns.PairGrid(result.posterior.iloc[:,observables_to_plot], corner=True, diag_sharey=False)
        gg.map_lower(sns.histplot, color=sns.color_palette()[4])
#g.map_upper(sns.kdeplot, shade=True, color=sns.color_palette()[0])
        gg.map_diag(sns.kdeplot, linewidth=2, shade=True, color=sns.color_palette()[9])
        for n,i in enumerate(observables_to_plot):
                ax=gg.axes[n][n]
                ax.axvline(x=map_parameters[i], ls='--', c=sns.color_palette()[9])
                ax.text(0,0.9,s= f'{map_parameters[i]:.3f}', transform=ax.transAxes)
        #os.makedirs(outdir+'/figures/', exist_ok = True)

        plt.tight_layout()
        plt.savefig(outdir+'/figures/'+'mix_params_posterior_'+label, dpi=100)
#plt.show()

#        os.makedirs(outdir+'/figures/', exist_ok = True)
        #plt.tight_layout()
        #fig.savefig(outdir+'/figures/'+'MAP_'+label, dpi=100)

    for i in range(0,n_obs):
        fig, axs = plt.subplots(1,1,figsize=(10,10))
        ax = axs
        #ax.plot(plot_g, mean[0][i,:].flatten(), label='posterior mean')
        # ax.errorbar(plot_g, m1_prediction[0][:,i].flatten(), 
        #             yerr=m1_prediction[1][:,i].flatten(), 
        #             label='Grad', alpha=0.2)
        # ax.errorbar(plot_g, m2_prediction[0][:,i].flatten(), 
        #             yerr=m1_prediction[1][:,i].flatten(),
        #             label='CE', alpha=0.2)
        ax.plot(plot_g, m1_prediction[0][:,i].flatten(),
                label=m1_name, alpha=0.8)
        ax.plot(plot_g, m2_prediction[0][:,i].flatten(), 
                label=m2_name, alpha=0.8)
        ax.fill_between(plot_g,per5[0][i,:].flatten(),per95[0][i,:].flatten(),color=sns.color_palette()[4], alpha=0.8, label='90% C.I.')
        ax.fill_between(plot_g,per20[0][i,:].flatten(),per80[0][i,:].flatten(), color=sns.color_palette()[4], alpha=0.5, label='60% C.I.')
        #ax.fill_between(plot_g,prior20[0][i,:].flatten(),prior80[0][i,:].flatten(),color=sns.color_palette()[2], alpha=0.5, label='60% C.I. Prior')
        ax.scatter(g,exp_data[0][:,i].flatten(), marker='x', label='experimental data')
        #ax.plot(plot_g, mean_prior[0][i,:].flatten(), label='prior mean')
        #ax.plot(plot_g, map_prediction[i,:].flatten(), label='MAP prediction', color='r')
        ax.set_xlabel('Centrality')
        ax.set_ylabel(obs_tex_labels[obs[i]])
        ax.legend()

        #ax_hist = axs.flatten()[1]
        #sns.histplot(data=result.posterior, x='step_0', kde=True, ax=ax_hist)
        #ax_hist.axvline(x = mix_model.map, color = 'r', label = f'MAP : {mix_model.map[0]}') 
        #ax_hist.legend()

        os.makedirs(outdir+'/figures/', exist_ok = True)
        plt.tight_layout()
        fig.savefig(outdir+'/figures/'+'posterior_'+label, dpi=100)

    


import time
obs_to_remove = ['dNch_deta', 'dET_deta', 'dN_dy_pion', 'dN_dy_kaon', 'dN_dy_proton', 'dN_dy_Lambda', 'dN_dy_Omega', 'dN_dy_Xi', 'mean_pT_pion', 'mean_pT_kaon', 'mean_pT_proton', 'pT_fluct']
obs_names = list(obs_cent_list['Pb-Pb-2760'].keys())
for i in obs_to_remove:
    obs_names.remove(i)
for obs in obs_names:
    print(f'For observable: {obs}')
    print(f'Start time {time.ctime()}')
    st = time.time()
    BMM_for_RHIC(m1_num=0, m2_num=3, 
                outdir='outdir/samba_bivaraite_grad_ptb_total_addstep', 
                label='addstep_mix'+'_'+obs, obs=[obs],
                g=np.linspace(0, 60, 20), 
                plot_g = np.linspace(0, 60, 20))
    print(f'Total time taken for {obs}; in hours {(st-time.time())/3600}')



