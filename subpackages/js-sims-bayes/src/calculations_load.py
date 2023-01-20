#!/usr/bin/env python3

from configurations import *
#import logging
import numpy as np

trimmed_model_data = {}
validation_data = {}
MAP_data = {}

for i, s in enumerate(system_strs):

    sdtype = [bayes_dtype[i]]

    try:
        Ndesign = SystemsInfo[s]['n_design']
        Ndelete = len(SystemsInfo[s]['design_remove_idx'])
        print("Loading {:s} main calculations from ".format(s) + SystemsInfo[s]['main_obs_file'])
        ds = np.fromfile(SystemsInfo[s]["main_obs_file"], dtype=sdtype)
        print("ds.shape = " + str(ds.shape))

        # handle some Nans
        #for pt in range(Ndesign): # loop over all design points
        #    for obs in active_obs_list[s]:
        #        values = np.array(ds[s][pt, idf][obs]['mean'])
        #        # delete Nan dataset
        #        isnan = np.isnan(values)
        #        if (np.sum(isnan) > 0) and (not pt in delete_design_pts_set):
        #            print("WARNING : FOUND NAN IN MODEL DATA : (design pt , obs)"\
        #                  +" = ( {:s} , {:s} )".format( str(pt), obs) )
        #            #ds[s][pt, idf][obs]['mean'][isnan] = np.mean(values[np.logical_not(isnan)])
        #            #transforming yield related observables
        #            is_mult = ('dN' in obs) or ('dET' in obs)
        #            if is_mult and transform_multiplicities:
        #                ds[s][pt, idf][obs]['mean'] = np.log(1.0 + values)

        if Ndelete > 0:
            print("Design points which will be deleted from training : " + str( SystemsInfo[s]["design_remove_idx"] ) )
            trimmed_model_data[s] = np.delete(ds[s], SystemsInfo[s]["design_remove_idx"], 0)
        else:
            print("No design points will be deleted from training")
            trimmed_model_data[s] = ds[s]

        #load the validation model calculations
        if validation:
            if pseudovalidation:
                validation_data[s] = trimmed_model_data[s]
            elif crossvalidation:
                validation_data[s] = ds[s]

            else:
                print("Loading {:s} validation calculations from ".format(s) + SystemsInfo[s]['validation_obs_file'])
                dsv = np.fromfile(SystemsInfo[s]["validation_obs_file"], dtype=sdtype)
                validation_data[s] = dsv[s]
            print("validation_data.shape = " + str(dsv.shape))
    except:
        print("WARNING! can not load model design calculations")

    #load the MAP calculations
    try:
        print("Loading {:s} MAP calculations from ".format(s) + SystemsInfo[s]['MAP_obs_file'])
        dsMAP = np.fromfile(SystemsInfo[s]["MAP_obs_file"], dtype=sdtype)
        MAP_data[s] = dsMAP[s]
        print("MAP_data.shape = " + str(dsMAP.shape))
    except:
        print("No MAP calculations found for system " + s)
