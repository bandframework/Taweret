calibration_jetscape_grad_or_ptb_product_likelihood.py has BMM method with "calibrate_model_1". The data is saved in outdir/calibration_grad_20000_product_likelihood. Things can be easily change in this file so we calibrate model_2 instead...

calibration_jetscape_grad_ptb_addstepasym.py has BMM method addstepasym mixing. The data is saved in "'outdir/calibration_grad_ptb_addstepasym_20000_10_exp'". addstepasym_grad_ptb_mixing_and_calibration.ipynb notebook can be used to regenrate plots using the saved data. 

We need to create a new file/notebook which use the calibrated models to do BMM and generate plots.
We start with calibrated_jetscape_grad_ptb_addstepasym.py file. Now need to modify it. 

Then start working on the novel BMM method which consider correlated data to BMM.


