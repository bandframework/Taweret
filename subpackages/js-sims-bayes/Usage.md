## Configurations

First see the parameters located in `src/configurations.py`. These set the number of design points, principal components, and other 'global' settings. 

## Generating Design

 To generate the design points: `./src/design.py design_pts`

This will create a directory called `design_pts`. 

To make a plot checking the design prior for all observables:

```./src/bayes_plot.py plots/obs_prior.png```

To generate a plot of the parameter values of all design points:

``` ./src/bayes_plot.py plots/param_prior.png```

## Running Events

Now copy `design_pts` to `sims_scripts/input_config/design_pts` on stampede2. 

Then look the script `submit_launcher_design_new_norm`, and change the parameters controlling number of design points and number of events per design point appropriately.

Submit the job : `sbatch submit_launcher_design_new_norm`

## Event Averaging

Once the events have finished, we have the script `src/calculations_average_obs.py` which can perform the event averaging. Make sure that the parameter `run_id` in `configurations.py` is set to match the name of the folder where you will store the events. Suppose that `run_id = my_events` , and the directory `model_calculations/my_events` exists. 

One should make two subdirectores `my_events/Events` and `my_events/Obs`

The raw event files before taking centrality bin averages will be stored in `Events/main`. Suppose we have 50 design points. Then, all of the events binary files generated for design point 0 should be catted together and stored in `0.dat`, and similarly for all other design points.

After this, there should exist `main/0.dat` , `main/1.dat` , ... , `main/49.dat` .

Now, one can run `./src/calculations_average_obs.py` to perform the centrality averaging of all events. 

## Building Emulator

To build the emulator: 

```./src/emulator.py --retrain --npc 10 --nrestarts 4```

This will build the emulator and store it as a dill file.

## Validating Emulator

To validate the emulator using a validation data set:

```./src/emulator_load_and_validate.py```

## MCMC

To run the MCMC for parameter estimation:

```./src/bayes_mcmc.py 2000 --nwalkers 100 --nburnsteps 500```

If a file exists `mcmc/chain.hdf` then the exisiting chain will be reused. To start a new chain, make sure that this file does not exist. 

To plot the estimation of parameters:

```./src/bayes_plot.py plots/diag_posterior.png```

To plot the entire posterior of all parameters:

```./src/bayes_plot.py plots/posterior.png```

To plot the posterior of viscosities:

```./src/bayes_plot.py plots/viscous_posterior.png```

To plot the emulator prediction using best fit parameters against data (or pseudodata):

```./src/bayes_plot.py plots/obs_validation```

