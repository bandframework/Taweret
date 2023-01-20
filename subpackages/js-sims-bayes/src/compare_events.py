#!/usr/bin/env python3

from configurations import *
import numpy as np

model_data_1 = {}
file1 = 'model_calculations/MAP/Grad/Obs/obs_Pb-Pb-2760_default.dat'
model_data_2 = {}
file2 = 'model_calculations/MAP/Grad/Obs/obs_Pb-Pb-2760_2x_bulk_relax_time.dat'

for i, s in enumerate(system_strs):
    sdtype = [bayes_dtype[i]]

    #load the model calculations
    print("Loading {:s} calculations")
    try:
        ds1 = np.fromfile(file1, dtype=sdtype)
        ds2 = np.fromfile(file2, dtype=sdtype)
        model_data_1[s] = ds1[s]
        model_data_2[s] = ds2[s]
        print("model_data_1.shape = " + str(ds1.shape))
        print("model_data_2.shape = " + str(ds2.shape))
    except:
        print("No model calculations found for system " + s)
