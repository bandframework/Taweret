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

# Toy models from SAMBA
#from Taweret.models import jetscape_sims_models as sims
#from Taweret.core.base_model import BaseModel

# Mixing method
#from Taweret.mix.bivariate_linear import BivariateLinear as BL

obs_to_remove = ['dN_dy_Lambda', 'dN_dy_Omega', 'dN_dy_Xi']

from Taweret.models import jetscape_sims_models as sims

m1 = sims.jetscape_models_pb_pb_2760(fix_MAP=True,model_num=0, obs_to_remove=obs_to_remove)
m2 = sims.jetscape_models_pb_pb_2760(fix_MAP=True,model_num=3, obs_to_remove=obs_to_remove)
exp = sims.exp_data()
g = np.linspace(0, 60, 10)

exp_data= exp.evaluate(g,obs_to_remove=obs_to_remove)

from Taweret.core.base_model import BaseModel

from Taweret.mix.bivariate_linear import BivariateLinear as BL

models= {'Grad':m1,'PTB':m2}
mix_model = BL(models_dic=models, method='addstepasym')

#uncoment to change the prior from the default
priors = bilby.core.prior.PriorDict()
priors['addstepasym_0'] = bilby.core.prior.Uniform(0, 60, name="addstepasym_0")
priors['addstepasym_1'] = bilby.core.prior.Uniform(0, 60, name="addstepasym_1")
priors['addstepasym_2'] = bilby.core.prior.Uniform(0, 1, name="addstepasym_2")
mix_model.set_prior(priors)

#result = mix_model.train(x_exp=g, y_exp=exp_data[0], y_err=exp_data[1], outdir = 'outdir/samba_bivaraite', label='step_mix')
outdir = 'outdir/calibrated_grad_ptb_addstepasym'
result = mix_model.train(x_exp=g, y_exp=exp_data[0], y_err=exp_data[1], outdir = outdir, label='addstepasym_mix', load_previous=True,)
#kwargs_for_sampler=sampler_args)

model_param_dsgn = ['$\\theta_0$', 
                    '$\\theta_1$',
                    '$\\alpha$',]
#     '$N$[$2.76$TeV]',
#  '$p$',
#  '$\\sigma_k$',
#  '$w$ [fm]',
#  '$d_{\\mathrm{min}}$ [fm]',
#  '$\\tau_R$ [fm/$c$]',
#  '$\\alpha$',
#  '$T_{\\eta,\\mathrm{kink}}$ [GeV]',
#  '$a_{\\eta,\\mathrm{low}}$ [GeV${}^{-1}$]',
#  '$a_{\\eta,\\mathrm{high}}$ [GeV${}^{-1}$]',
#  '$(\\eta/s)_{\\mathrm{kink}}$',
#  '$(\\zeta/s)_{\\max}$',
#  '$T_{\\zeta,c}$ [GeV]',
#  '$w_{\\zeta}$ [GeV]',
#  '$\\lambda_{\\zeta}$',
#  '$b_{\\pi}$',
#  '$T_{\\mathrm{sw}}$ [GeV]']

bound_array = []
for param_name in mix_model.prior.keys():
    prior = mix_model.prior[param_name]
    a,b = prior.minimum, prior.maximum
    bound_array.append((a,b))
print(bound_array)

# If false, do not try to find the MAP value and load it from a saved file.
find_map_param = True
cal_name = '/map_values'
from scipy import optimize
if find_map_param == True:
    #bounds=[(a,b) for (a,b) in zip(design_min,design_max)]
    bounds= bound_array
    print(bounds)
    #x0 = [a+b/2 for a,b in bounds]
    x0 = mix_model.map
    #rslt = optimize.differential_evolution(lambda x: -cal.theta.lpdf(theta=np.array(x).reshape(-1,15)).flatten(),
    #                                        bounds=bounds,
    #                                       maxiter=100000,
    #                                        tol=1e-9,
    #                                        disp=True)
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds,"tol":1e-11}
    #rslt=optimize.basinhopping(lambda x: -log_posterior(x), x0,niter=100,minimizer_kwargs=minimizer_kwargs)
    rslt=optimize.basinhopping(lambda x: -mix_model.mix_loglikelihood(x[0:3],[],
                                                                      x_exp=g, 
                                                                      y_exp=exp_data[0], 
                                                                      y_err=exp_data[1]), 
                               x0,niter=100,minimizer_kwargs=minimizer_kwargs)
    map_values = rslt.x
    np.save(outdir+cal_name, map_values)
else:
    map_values = np.load(outdir+cal_name+'.npy')
    
print('Optimization MAP log likelihood')
print(mix_model.mix_loglikelihood(map_values[0:3], [], x_exp=g, y_exp=exp_data[0], y_err=exp_data[1]))
print('Sort MAP log likelihood')
mix_model.mix_loglikelihood(mix_model.map[0:3], [], x_exp=g, y_exp=exp_data[0], y_err=exp_data[1])

prior_df_temp = pd.DataFrame(mix_model.prior.sample(result.posterior.shape[0]))
prior_df = pd.DataFrame(prior_df_temp.values, columns=model_param_dsgn)

samples_df = pd.DataFrame(result.posterior.values[:,:-2],columns=model_param_dsgn)


#from plotting import plot_corner_viscosity

#plot_corner_viscosity(samples_df,prior_df, outdir, n_samples=result.posterior.shape[0], prune=10, MAP=map_values, closure=None)

#from plotting import plot_corner_no_viscosity
#plot_corner_no_viscosity(samples_df,prior_df, outdir, n_samples=result.posterior.shape[0], prune=10, MAP=map_values, closure=None)

#from plotting import plot_bulk, plot_shear
#plot_shear(samples_df, prior_df, outdir, n_samples=result.posterior.shape[0], prune=10, MAP=map_values, closure=None, ax= None, legend=False)
#plot_bulk(samples_df, prior_df, outdir, n_samples=result.posterior.shape[0], prune=10, MAP=map_values, closure=None, ax= None, legend=False)

_,mean_w_prior,CI_w_prior, _ = mix_model.predict_weights(g, CI=[5,20,80,95], samples=prior_df.values)
_,mean_w,CI_w, _ = mix_model.predict_weights(g, CI=[5,20,80,95])

per5_w, per20_w, per80_w, per95_w = CI_w
prior5_w, prior20_w, prior80_w, prior95_w = CI_w_prior

#plt.show()
#map_values= mix_model.map
map_values = map_values
w1,_ = mix_model.evaluate_weights(map_values[0:3],g)
#w1,_ = mix_model.evaluate_weights(np.array([0.2, 0]),g)

fig, ax = plt.subplots(figsize=(10,10))
#ax.set_title('MAP')
ax.plot(g, w1, label = 'MAP ' + str([f'{mp:.1f}' for mp in map_values[0:3]]))

ax.fill_between(g,per5_w.flatten(),per95_w.flatten(),color=sns.color_palette()[4], alpha=0.4, label='90% C.I.')
ax.fill_between(g,per20_w.flatten(),per80_w.flatten(), color=sns.color_palette()[4], alpha=0.3, label='60% C.I.')
#ax.fill_between(g,prior20_w.flatten(),prior80_w.flatten(),color=sns.color_palette()[2], alpha=0.5, label='60% C.I. Prior')
    
ax.set_xlabel('Centrality')
ax.set_ylabel('Model_1 weight')
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig(outdir+'/figures/'+'MAP_mixing_function_model_1', dpi=100)
#plt.show()

_,mean_prior,CI_prior, _ = mix_model.prior_predict(g, CI=[5,20,80,95])
_,mean,CI, _ = mix_model.predict(g, CI=[5,20,80,95], nthin=10)

per5, per20, per80, per95 = CI
prior5, prior20, prior80, prior95 = CI_prior

print(f'Map values {map_values}')

map_prediction = mix_model.evaluate(map_values[0:3], g, [map_values[3:]])


map_parameters=map_values.flatten()
sns.set_context('paper')
sns.set_palette('bright')
observables_to_plot=[0, 1, 2]
gg = sns.PairGrid(samples_df.iloc[:,observables_to_plot], corner=True, diag_sharey=False)
gg.map_lower(sns.histplot, color=sns.color_palette()[4])
#g.map_upper(sns.kdeplot, shade=True, color=sns.color_palette()[0])
gg.map_diag(sns.kdeplot, linewidth=2, shade=True, color=sns.color_palette()[9])
for n,i in enumerate(observables_to_plot):
    ax=gg.axes[n][n]
    ax.axvline(x=map_parameters[i], ls='--', c=sns.color_palette()[9])
    ax.text(0,0.9,s= f'{map_parameters[i]:.3f}', transform=ax.transAxes)

plt.tight_layout()
plt.savefig(outdir+'/figures/'+'posterior_mix', dpi=100)

#g = np.linspace(0, 60, 20)
#plot_g = np.linspace(0.0,60,100)
map_mix = map_values[3:]
m1_prediction = m1.evaluate(g, map_mix)
m2_prediction = m2.evaluate(g, map_mix)
#true_output = truth.evaluate(plot_g)
exp_data= exp.evaluate(g,obs_to_remove=obs_to_remove)

obs_names = list(obs_cent_list['Pb-Pb-2760'].keys())
for i in obs_to_remove:
     obs_names.remove(i)
print(obs_names)

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





fig, axs = plt.subplots(3,4, figsize=(40,30))
sns.set_context('poster')
for i in range(0,12):
    ax_f= axs.flatten()[i]
    ax_f.errorbar(g, m1_prediction[0][:,i].flatten(), 
                yerr=m1_prediction[1][:,i].flatten(), 
                label='Grad', alpha=0.5)
    ax_f.errorbar(g, m2_prediction[0][:,i].flatten(), 
                yerr=m1_prediction[1][:,i].flatten(),
                label='PTB', alpha=0.5)
    # ax_f.plot(plot_g, m1_prediction[0][:,i].flatten(),
    #             label='Grad', alpha=0.8)
    # ax_f.plot(plot_g, m2_prediction[0][:,i].flatten(), 
    #             label='PTB', alpha=0.8)
    #ax_f.plot(g, mean[0][i,:].flatten(), label='Mean BMM')
    ax_f.plot(g, map_prediction[i,:].flatten(), label='MAP', color='k')
    #ax_f.plot(plot_g, true_output[0], label='truth')
    ax_f.errorbar(g, exp_data[0][:,i].flatten(), 
                yerr=exp_data[1][:,i].flatten(),
                #marker='x',
                fmt='o',
                label='experimental data',
                color='r')
    #ax_f.scatter(g,exp_data[0][:,i].flatten(), marker='x', label='experimental data', color='r')
    ax_f.set_xlabel('Centrality')
    #ax_f.set_ylim(1.2,3.2)
    ax_f.set_ylabel(obs_tex_labels[obs_names[i]])
    if i ==0:
        ax_f.legend()
plt.tight_layout()
fig.savefig(outdir+'/figures/'+'MAP_', dpi=100)



sns.set_context('poster')
fig, axs = plt.subplots(3, 4, figsize=(40,30))
for i in range(0,12):
    ax = axs.flatten()[i]
    #fig, ax = plt.subplots(figsize=(10,10))
    #ax.plot(plot_g, mean[0][i,:].flatten(), label='posterior mean')
    # ax_f.errorbar(plot_g, m1_prediction[0][:,i].flatten(), 
    #             yerr=m1_prediction[1][:,i].flatten(), 
    #             label='Grad', alpha=0.2)
    # ax_f.errorbar(plot_g, m2_prediction[0][:,i].flatten(), 
    #             yerr=m1_prediction[1][:,i].flatten(),
    #             label='PTB', alpha=0.2)
#     ax.plot(g, m1_prediction[0][:,i].flatten(),
#                 label='Grad', alpha=0.8)
#     ax.plot(g, m2_prediction[0][:,i].flatten(), 
#                 label='PTB', alpha=0.8)
    ax.fill_between(g,per5[0][i,:].flatten(),per95[0][i,:].flatten(),color=sns.color_palette()[4], alpha=0.8, label='90% C.I.')
    ax.fill_between(g,per20[0][i,:].flatten(),per80[0][i,:].flatten(), color=sns.color_palette()[4], alpha=0.5, label='60% C.I.')
    ax.fill_between(g,prior20[0][i,:].flatten(),prior80[0][i,:].flatten(),color=sns.color_palette()[2], alpha=0.2, label='60% C.I. Prior')
#    ax.scatter(g,exp_data[0][:,i].flatten(), marker='x', label='experimental data')
    ax.errorbar(g, exp_data[0][:,i].flatten(), 
                yerr=exp_data[1][:,i].flatten(),
                #marker='x',
                fmt='o',
                label='experimental data',
                color='r')
    #ax.plot(plot_g, mean_prior[0][i,:].flatten(), label='prior mean')
    ax.plot(g, map_prediction[i,:].flatten(), label='MAP prediction', color='k')
    ax.set_xlabel('Centrality')
    #ax_f.set_ylim(1.2,3.2)
    ax.set_ylabel(obs_tex_labels[obs_names[i]])
    if i==0:
        ax.legend()
plt.tight_layout()
fig.savefig(outdir+'/figures/'+'posterior_predict_', dpi=100)







# sns.set_context('poster')
# result.plot_corner()
# plt.savefig(outdir+'/figures/'+'corner_', dpi=100)
# #result.posterior

# _,mean_prior,CI_prior, _ = mix_model.prior_predict(g, CI=[5,20,80,95])
# _,mean,CI, _ = mix_model.predict(g, CI=[5,20,80,95])

# per5, per20, per80, per95 = CI
# prior5, prior20, prior80, prior95 = CI_prior

# print(f'Map values {mix_model.map}')

# map_prediction = mix_model.evaluate(mix_model.map[0:3], g, [mix_model.map[3:-1]])

# #sns.pairplot(result.posterior[['addstep_0','addstep_1','addstep_2']], kind='kde', diag_kind='kde')
# map_parameters=mix_model.map.flatten()
# sns.set_palette('bright')
# observables_to_plot=[0, 1, 2]
# gg = sns.PairGrid(result.posterior.iloc[:,observables_to_plot], corner=True, diag_sharey=False)
# gg.map_lower(sns.histplot, color=sns.color_palette()[4])
# #g.map_upper(sns.kdeplot, shade=True, color=sns.color_palette()[0])
# gg.map_diag(sns.kdeplot, linewidth=2, shade=True, color=sns.color_palette()[9])
# for n,i in enumerate(observables_to_plot):
#     ax=gg.axes[n][n]
#     ax.axvline(x=map_parameters[i], ls='--', c=sns.color_palette()[9])
#     ax.text(0,0.9,s= f'{map_parameters[i]:.3f}', transform=ax.transAxes)

# plt.tight_layout()
# plt.savefig(outdir+'/figures/'+'posterior_', dpi=100)
# #plt.show()


# w1,_ = mix_model.evaluate_weights(mix_model.map.flatten()[0:3],g)
# #w1,_ = mix_model.evaluate_weights(np.array([0.2, 0]),g)

# fig, ax = plt.subplots(figsize=(10,10))
# #ax.set_title('MAP')
# ax.plot(g, w1, label = 'MAP ' + str([f'{mp:.1f}' for mp in mix_model.map.flatten()]))
# ax.set_xlabel('Centrality')
# ax.set_ylabel('Model_1 weight')
# ax.legend(loc='upper left')

# plt.tight_layout()
# plt.savefig(outdir+'/figures/'+'MAP_mixing_function_', dpi=100)
# #plt.show()

# obs_tex_labels = {
#                     'dNch_deta' : r'$dN_{ch}/d\eta$',
#                     'dN_dy_pion' : r'$dN_{\pi}/dy$',
#                     'dN_dy_kaon' : r'$dN_{k}/dy$',
#                     'dN_dy_proton' : r'$dN_{p}/dy$',
#                     'dN_dy_Lambda' : r'$dN_{\Lambda}/dy$',
#                     'dN_dy_Omega' : r'$dN_{\Omega}/dy$',
#                     'dN_dy_Xi' : r'$dN_{\Xi}/dy$',
#                     'dET_deta' : r'$dE_{T}/d\eta$',
#                     'mean_pT_pion' : r'$\langle p_T \rangle _{\pi}$',
#                     'mean_pT_kaon' : r'$\langle p_T \rangle _{k}$',
#                     'mean_pT_proton' : r'$\langle p_T \rangle _{p}$',
#                     'pT_fluct' : r'$\delta p_T / \langle p_T \rangle$',
#                     'v22' : r'$v_2\{2\}$',
#                     'v32' : r'$v_3\{2\}$',
#                     'v42' : r'$v_4\{2\}$',
# }






# fig, axs = plt.subplots(3,4, figsize=(40,30))
# sns.set_context('poster')
# for i in range(0,12):
#     ax_f= axs.flatten()[i]
#     ax_f.errorbar(g, m1_prediction[0][:,i].flatten(), 
#                 yerr=m1_prediction[1][:,i].flatten(), 
#                 label='Grad', alpha=0.5)
#     ax_f.errorbar(g, m2_prediction[0][:,i].flatten(), 
#                 yerr=m1_prediction[1][:,i].flatten(),
#                 label='PTB', alpha=0.5)
#     # ax_f.plot(plot_g, m1_prediction[0][:,i].flatten(),
#     #             label='Grad', alpha=0.8)
#     # ax_f.plot(plot_g, m2_prediction[0][:,i].flatten(), 
#     #             label='PTB', alpha=0.8)
#     ax_f.plot(g, mean[0][i,:].flatten(), label='Mean BMM')
#     ax_f.plot(g, map_prediction[i,:].flatten(), label='MAP', color='k')
#     #ax_f.plot(plot_g, true_output[0], label='truth')
#     ax_f.scatter(g,exp_data[0][:,i].flatten(), marker='x', label='experimental data', color='r')
#     ax_f.set_xlabel('Centrality')
#     #ax_f.set_ylim(1.2,3.2)
#     ax_f.set_ylabel(obs_tex_labels[obs_names[i]])
#     if i ==0:
#         ax_f.legend()
# plt.tight_layout()
# fig.savefig(outdir+'/figures/'+'MAP_', dpi=100)

# sns.set_context('poster')
# fig, axs = plt.subplots(3, 4, figsize=(40,30))
# for i in range(0,12):
#     ax = axs.flatten()[i]
#     #fig, ax = plt.subplots(figsize=(10,10))
#     #ax.plot(plot_g, mean[0][i,:].flatten(), label='posterior mean')
#     # ax_f.errorbar(plot_g, m1_prediction[0][:,i].flatten(), 
#     #             yerr=m1_prediction[1][:,i].flatten(), 
#     #             label='Grad', alpha=0.2)
#     # ax_f.errorbar(plot_g, m2_prediction[0][:,i].flatten(), 
#     #             yerr=m1_prediction[1][:,i].flatten(),
#     #             label='PTB', alpha=0.2)
#     ax.plot(g, m1_prediction[0][:,i].flatten(),
#                 label='Grad', alpha=0.8)
#     ax.plot(g, m2_prediction[0][:,i].flatten(), 
#                 label='PTB', alpha=0.8)
#     ax.fill_between(g,per5[0][i,:].flatten(),per95[0][i,:].flatten(),color=sns.color_palette()[4], alpha=0.8, label='90% C.I.')
#     ax.fill_between(g,per20[0][i,:].flatten(),per80[0][i,:].flatten(), color=sns.color_palette()[4], alpha=0.5, label='60% C.I.')
#     #ax.fill_between(plot_g,prior20[0][i,:].flatten(),prior80[0][i,:].flatten(),color=sns.color_palette()[2], alpha=0.2, label='60% C.I. Prior')
#     ax.scatter(g,exp_data[0][:,i].flatten(), marker='x', label='experimental data')
#     #ax.plot(plot_g, mean_prior[0][i,:].flatten(), label='prior mean')
#     ax.plot(g, map_prediction[i,:].flatten(), label='MAP prediction', color='r')
#     ax.set_xlabel('Centrality')
#     #ax_f.set_ylim(1.2,3.2)
#     ax.set_ylabel(obs_tex_labels[obs_names[i]])
#     if i==0:
#         ax.legend()
# plt.tight_layout()
# fig.savefig(outdir+'/figures/'+'posterior_predict_', dpi=100)


