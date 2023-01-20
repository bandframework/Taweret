#!/usr/bin/env python3

from configurations import *
import logging
import numpy as np
import scipy.stats

print("Loading model calculations from " + f_obs_main)
model_data = np.fromfile(f_obs_main, dtype=bayes_dtype)
print("model_data.shape = " + str(model_data.shape))


# Loop over all observables
events_nan_all_df=[]
events_nan=[]
events_mean_pT_odd=[]
events_mean_pT_increasing=[]

for design_pt in range(n_design_pts_main): # loop over all design points
    system_str = system_strs[0]
    for obs in active_obs_list[system_str]:
        #print(model_data[system_str][:,:][obs]['mean'].shape)

        #
        nan_all_df=True
        nan_any_df=False
        mean_pT_odd_any_df=False
        mean_pT_increasing_any_df=False

        # Get centrality bins
        cent_list=obs_cent_list[system_str][obs][:]
        cent_list_mid=np.array([(a+b)/2 for (a,b) in  cent_list])

        # Nevermind NaN's in very peripheral events
        nan_centrality_cut=60

        for df_index in range(number_of_models_per_run):
        #for df_index in [idf]:
            values = np.array( model_data[system_str][design_pt, df_index][obs]['mean'])

            cent_sel_bool=cent_list_mid<nan_centrality_cut
            if (np.all(cent_sel_bool)):
                nan_cent_i_max=len(cent_list_mid)
            else:
                nan_cent_i_max=np.argmin(cent_sel_bool)

            nan_vals=values[:nan_cent_i_max]

            # Check for NaN's in any observables
            isnan = np.isnan(nan_vals)
            if (isnan.any())and(not obs == "v42"):
                #print("Problem with observable ",obs," of design point ",design_pt,"(system=",system_str ,", idf=",df_index, ":", values)
                nan_any_df=True
            else:
                nan_all_df=False

            # Check for non-monotonic mean p_T
            if (obs == 'mean_pT_pion'):
                max_variation=.07
                # Restrict test to not-too-peripheral centralities
                centrality_cut=80
                cent_i_max=np.argmin(cent_list_mid<centrality_cut)-1

                cent_mid_vals=np.array(cent_list_mid[:cent_i_max])
                mean_pT_vals=values[:cent_i_max]

                #print(mean_pT_vals)
                #check if mean pT is decreasing with centrality
                mean_pT_vals_sorted = sorted(mean_pT_vals, reverse = True)
                """
                if ( (mean_pT_vals == mean_pT_vals_sorted).all() ):
                    pass
                else :
                    mean_pT_increasing_any_df=True
                """
                if ( mean_pT_vals[-1] > 1.05 * mean_pT_vals[-2] ):
                    mean_pT_increasing_any_df=True
                else :
                    pass



                _,_,rvalue,pvalue,_=scipy.stats.linregress(cent_mid_vals, mean_pT_vals)
                #print(mean_pT_vals,rvalue, pvalue)
                std_over_mean=(np.std(mean_pT_vals)/np.mean(mean_pT_vals))
                rel_median_over_max=np.abs(np.max(mean_pT_vals)/np.mean(mean_pT_vals)-1)
                #if (np.power(np.mean(np.power(abs(mean_pT_vals - mean_pT_vals.mean()),power_test)),1./power_test)/np.mean(mean_pT_vals)>0.1):
                #if (std_over_mean>0.01)and(np.abs(rvalue)<0.95):
                #    print(design_pt,mean_pT_vals,rvalue, pvalue, std_over_mean, rel_median_over_mean)

                #print(mean_pT_vals)
                #np.all(values[:central_bins_to_test] >= values[1:(central_bins_to_test+1)])
                #if (np.std(mean_pT_vals)/np.mean(mean_pT_vals) > max_variation):
                #if (np.std(mean_pT_vals)/np.mean(mean_pT_vals) > 0.05)and(np.abs(np.median(mean_pT_vals)/np.mean(mean_pT_vals)-1) > 0.02):
                #power_test=12
#                if (np.abs(np.median(mean_pT_vals)/np.mean(mean_pT_vals)-1) > 0.02):
                #if (np.power(np.mean(np.power(abs(mean_pT_vals - mean_pT_vals.mean()),power_test)),1./power_test)/np.mean(mean_pT_vals)>0.1):
                #if (np.correlate(mean_pT_vals[0:-1], mean_pT_vals[1:], mode='valid')/np.sum(mean_pT_vals*mean_pT_vals) < .795):
                #if (np.abs(np.median(mean_pT_vals)/np.mean(mean_pT_vals)-1) > max_variation):
                if (np.abs(rvalue) < .95)and(std_over_mean > 0.01)and(rel_median_over_max>0.05):
                    mean_pT_odd_any_df=True



        if (nan_all_df):
            events_nan_all_df.append(design_pt)

        if (nan_any_df):
            events_nan.append(design_pt)

        if (mean_pT_odd_any_df):
            events_mean_pT_odd.append(design_pt)

        if (mean_pT_increasing_any_df):
            events_mean_pT_increasing.append(design_pt)

        #if np.sum(isnan) > 0:
        # delete Nan dataset
        #isnan = np.isnan(values)
        #if np.sum(isnan) > 0:
        #    model_data[system_str][pt, idf][obs]['mean'][isnan] = np.mean(values[np.logical_not(isnan)])
        #if 'dN' in obs or 'dET' in obs:
        #    model_data[system_str][pt, idf][obs]['mean'] = np.log(1+values)
print("------------------------------------------------------------------------")
print("Events with NaN (centrality<",nan_centrality_cut,"): ",sorted(set(events_nan)))
print("Events with NaN for all 4 delta-f (centrality<",nan_centrality_cut,"): ",sorted(set(events_nan_all_df)))
print("Events with large fluctuations in pion mean p_T: ",sorted(set(events_mean_pT_odd)))
print("Events with non-decreasing pion mean pT: ",sorted(set(events_mean_pT_increasing)))
print("------------------------------------------------------------------------")
#
## things to drop for validation
#np.random.seed(1)
#delete_sets = []#np.random.choice(range(50), 10, replace=False)
#
#if len(delete_sets) > 0 :
#    print("Design points which will be deleted from training : " + str( np.sort( list(delete_sets) ) ) )
#    trimmed_model_data = np.delete(model_data, list(delete_sets), 0)
#else :
#    print("No design points will be deleted from training")
#    trimmed_model_data = model_data
