#
echo "Check if the mcmc/<chain-name>.hdf is the one you want to append the chain" 
#average the observables by centrality
#./src/calculations_average_obs.py

#train the emulator
./src/emulator.py --retrain --nrestarts 4

#perform MCMC
./src/bayes_mcmc.py 4000 --nwalkers 100 --nburnsteps 500


./src/bayes_plot.py plots/diag_pca.png # PCA Diagnostic, check uncorrelated
./src/bayes_plot.py plots/diag_emu.png # PCA Diagnostic, check how PCs respond to parameter changes
./src/bayes_plot.py plots/obs_prior.png # The prior against expt data
./src/bayes_plot.py plots/diag_posterior.png # The posterior of parameters
./src/bayes_plot.py plots/viscous_posterior.png # the posterior of eta/s and zeta/s
./src/bayes_plot.py plots/obs_validation.png # Samples from the emulator using MAP against experimental data (the fit)

#open plots/diag_pca.png
#open plots/diag_emu.png
#open plots/obs_prior.png
#open plots/diag_posterior.png
#open plots/viscous_posterior.png
open plots/obs_validation.png
