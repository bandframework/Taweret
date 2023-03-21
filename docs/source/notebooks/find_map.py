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

obs_to_remove = ['dN_dy_Lambda', 'dN_dy_Omega', 'dN_dy_Xi']

from Taweret.models import jetscape_sims_models as sims

m1 = sims.jetscape_models_pb_pb_2760(fix_MAP=False,model_num=0, obs_to_remove=obs_to_remove)
m2 = sims.jetscape_models_pb_pb_2760(fix_MAP=False,model_num=3, obs_to_remove=obs_to_remove)
exp = sims.exp_data()
g = np.linspace(0, 60, 20)

exp_data= exp.evaluate(g,obs_to_remove=obs_to_remove)

from Taweret.core.base_model import BaseModel

from Taweret.mix.bivariate_linear import BivariateLinear as BL

models= {'Grad':m1,'PTB':m2}
mix_model = BL(models_dic=models, method='addstepasym', nargs_model_dic={'Grad':17, 'PTB':17}, same_parameters = True)

#uncoment to change the prior from the default
priors = bilby.core.prior.PriorDict()
priors['addstepasym_0'] = bilby.core.prior.Uniform(0, 60, name="addstepasym_0")
priors['addstepasym_1'] = bilby.core.prior.Uniform(0, 60, name="addstepasym_1")
priors['addstepasym_2'] = bilby.core.prior.Uniform(0, 1, name="addstepasym_2")
mix_model.set_prior(priors)

#result = mix_model.train(x_exp=g, y_exp=exp_data[0], y_err=exp_data[1], outdir = 'outdir/samba_bivaraite', label='step_mix')
outdir = 'outdir/calibration_grad_ptb_addstepasym'
result = mix_model.train(x_exp=g, y_exp=exp_data[0], y_err=exp_data[1], outdir = outdir, label='addstepasym_mix', load_previous=True,)
#kwargs_for_sampler=sampler_args)

#sns.set_context('poster')
#result.plot_corner()
#plt.savefig(outdir+'/figures/'+'corner_', dpi=100)


#Model parameter names in Latex compatble form
model_param_dsgn = ['$\\theta$', 
                    '$\\gamma$',
                    '$\\beta$',
    '$N$[$2.76$TeV]',
 '$p$',
 '$\\sigma_k$',
 '$w$ [fm]',
 '$d_{\\mathrm{min}}$ [fm]',
 '$\\tau_R$ [fm/$c$]',
 '$\\alpha$',
 '$T_{\\eta,\\mathrm{kink}}$ [GeV]',
 '$a_{\\eta,\\mathrm{low}}$ [GeV${}^{-1}$]',
 '$a_{\\eta,\\mathrm{high}}$ [GeV${}^{-1}$]',
 '$(\\eta/s)_{\\mathrm{kink}}$',
 '$(\\zeta/s)_{\\max}$',
 '$T_{\\zeta,c}$ [GeV]',
 '$w_{\\zeta}$ [GeV]',
 '$\\lambda_{\\zeta}$',
 '$b_{\\pi}$',
 '$T_{\\mathrm{sw}}$ [GeV]']


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
    rslt=optimize.basinhopping(lambda x: -mix_model.mix_loglikelihood(x[0:3],[x[3:]],
                                                                      x_exp=g, 
                                                                      y_exp=exp_data[0], 
                                                                      y_err=exp_data[1]), 
                               x0,niter=100,minimizer_kwargs=minimizer_kwargs)
    map_values = rslt.x
    np.save(outdir+cal_name, map_values)
else:
    map_values = np.load(outdir+cal_name+'.npy')


