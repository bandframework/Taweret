#!/usr/bin/env python3

import numpy as np
#import h5py
import sys, os, glob
# Input data format
from calculations_file_format_single_event import *
# Output data format
from configurations import *

def list2array(func):
        def func_wrapper(x, w):
                try:
                        x = np.array(x)
                        w = np.array(w)
                except:
                        raise ValueError("cannot interpret input as numpy array...")
                return func(x, w)
        return func_wrapper

def weighted_mean_std(x, w=None):
        if w is None:
                Neff = x.size
                mean = np.mean(x)
                std = np.std(x)/np.sqrt(Neff-1.+1e-9)
        else:
                Neff = np.sum(w)**2/np.sum(w**2)
                mean = np.average(x, weights=w)
                std = ( np.average((x-mean)**2, weights=w)/(Neff-1.+1e-9) ) **.5
        return mean, std

def weighted_mean_std_exclude_zeros(x, w=None):
        x = x[np.nonzero(x)]
        if w is None:
                Neff = x.size
                mean = np.mean(x)
                std = np.std(x)/np.sqrt(Neff-1.+1e-9)
        else:
                Neff = np.sum(w)**2/np.sum(w**2)
                mean = np.average(x, weights=w)
                std = ( np.average((x-mean)**2, weights=w)/(Neff-1.+1e-9) ) **.5
        return mean, std

def mean_std(x):
        Neff = x.shape[0]
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)/np.sqrt(Neff - 1. + 1e-9)
        return mean, std

def calculate_dNdeta(ds, exp, cen, idf):
        Ne = len(ds)
        cenM = np.mean(cen, axis=1)
        index = (cen/100.*Ne).astype(int)
        obs = np.zeros_like(cenM)
        obs_err = np.zeros_like(cenM)
        for i, (nl, nh) in enumerate(zip(index[:,0], index[:,1])):
                nh = np.max([nh, nl+1])
                obs[i], obs_err[i] = weighted_mean_std( ds[exp]['dNch_deta'][nl:nh, idf] )
        return {'Name': 'dNch_deta', 'cenM': cenM, 'pTM' : None,
                        'obs': obs, 'err': obs_err}


def calculate_dETdeta(ds, exp, cen, idf):
        Ne = len(ds)
        cenM = np.mean(cen, axis=1)
        index = (cen/100.*Ne).astype(int)
        obs = np.zeros_like(cenM)
        obs_err = np.zeros_like(cenM)
        for i, (nl, nh) in enumerate(zip(index[:,0], index[:,1])):
                obs[i], obs_err[i] = weighted_mean_std(ds[exp]['dET_deta'][nl:nh, idf])
        return {'Name': 'dNch_deta', 'cenM': cenM, 'pTM' : None,
                        'obs': obs, 'err': obs_err}

def calculate_dNdy(ds, exp, cen, idf):
        Ne = len(ds)
        cenM = np.mean(cen, axis=1)
        index = (cen/100.*Ne).astype(int)
        obs = {s: np.zeros_like(cenM) for (s, _) in species}
        obs_err = {s: np.zeros_like(cenM) for (s, _) in species}
        for (s, _) in species:
                for i, (nl, nh) in enumerate(zip(index[:,0], index[:,1])):
                        obs[s][i], obs_err[s][i] = weighted_mean_std(ds[exp]['dN_dy'][s][nl:nh, idf])
        return {'Name': 'dNch_deta', 'cenM': cenM, 'pTM' : None,
                        'obs': obs, 'err': obs_err}

def calculate_dNdpT(ds, exp, cen, idf, s):
        Ne = len(ds)
        cenM = np.mean(cen, axis=1)
        index = (cen/100.*Ne).astype(int)
        obs = np.zeros( (len(cenM), Qn_diff_NpT) )
        obs_err = np.zeros( (len(cenM), Qn_diff_NpT) )

        for i, (nl, nh) in enumerate(zip(index[:,0], index[:,1])):
            #we need to manually normalize the number of particles by the number of oversamples
            nos = ds[exp]['nsamples'][nl:nh, idf]

            #normalize by the number of oversamples
            dN_dpT = np.divide(  ds['d_flow_pid'][s]['N'][nl:nh, idf].T, nos ).T
            obs[i,:], obs_err[i,:] = mean_std(dN_dpT)

        return {'Name': 'dN_dpT', 'cenM': cenM, 'pTM' : None,
                        'obs': obs, 'err': obs_err}

def calculate_mean_pT(ds, exp, cen, idf):
        Ne = len(ds)
        cenM = np.mean(cen, axis=1)
        index = (cen/100.*Ne).astype(int)
        obs = {s: np.zeros_like(cenM) for (s, _) in species}
        obs_err = {s: np.zeros_like(cenM) for (s, _) in species}
        for (s, _) in species:
                for i, (nl, nh) in enumerate(zip(index[:,0], index[:,1])):
                        #old procedure doesn't make sense for particles with less than yield one per event
                        #obs[s][i], obs_err[s][i] = weighted_mean_std(ds[exp]['mean_pT'][s][nl:nh, idf])
                        #new procedure only averages over pT of particles found 
                        obs[s][i], obs_err[s][i] = weighted_mean_std_exclude_zeros(ds[exp]['mean_pT'][s][nl:nh, idf])
        return {'Name': 'dNch_deta', 'cenM': cenM, 'pTM' : None,
                        'obs': obs, 'err': obs_err}

def calculate_mean_pT_fluct(ds, exp, cen, idf):

        Ne = len(ds)
        cenM = np.mean(cen, axis=1)
        index = (cen/100.*Ne).astype(int)
        obs = np.zeros_like(cenM)
        obs_err = np.zeros_like(cenM)
        for (s, _) in species:

                for i, (nl, nh) in enumerate(zip(index[:,0], index[:,1])):
                        N = ds[exp]['pT_fluct_chg']['N'][nl:nh, idf]
                        sum_pT = ds[exp]['pT_fluct_chg']['sum_pT'][nl:nh, idf]
                        sum_pTsq = ds[exp]['pT_fluct_chg']['sum_pT2'][nl:nh, idf]


                        Npairs = .5*N*(N - 1)

                        if ( sum_pT.sum() > 0. ):
                            M = sum_pT.sum() / N.sum()
                            # This is equivalent to the sum over pairs in Eq. (2).  It may be derived
                            # by using that, in general,
                            #
                            #   \sum_{i,j>i} a_i a_j = 1/2 [(\sum_{i} a_i)^2 - \sum_{i} a_i^2].
                            #
                            # That is, the sum over pairs (a_i, a_j) may be re-expressed in terms of
                            # the sum of a_i and sum of squares a_i^2.  Applying this to Eq. (2) and
                            # collecting terms yields the following expression.
                            x = (.5*(sum_pT**2 - sum_pTsq) - M*(N - 1)*sum_pT + M**2*Npairs)/Npairs
                            meanC, stdC = weighted_mean_std(x, Npairs)
                            obs[i] = np.sqrt(meanC)/M
                            obs_err[i] = stdC*.5/np.sqrt(meanC)/M
                        else :
                            obs[i] = 0.
                            obs_err[i] = 0.

        return {'Name': 'dNch_deta', 'cenM': cenM, 'pTM' : None,
                        'obs': obs, 'err': obs_err}


def calculate_vn(ds, exp, cen, idf):
        @list2array
        def obs_and_err(qn, m):
                w = m*(m-1.) # is this P_{M,2} in notation of Jonah's Thesis
                if w.sum() == 0.:
                    return 0., 0.
                cn2 = (np.abs(qn)**2 - m)/w # is this is <2> in Jonah's thesis (p.27)
                avg_cn2, std_avg_cn2 = weighted_mean_std(cn2, w)
                if avg_cn2 < 0. :
                    vn = -np.sqrt(-avg_cn2)
                else :
                    vn = np.sqrt(avg_cn2)
                vn_err = std_avg_cn2/2./vn
                return vn, vn_err
        Ne = len(ds)
        cenM = np.mean(cen, axis=1)
        index = (cen/100.*Ne).astype(int)

        obs = np.zeros([len(cenM), Nharmonic])
        obs_err = np.zeros([len(cenM), Nharmonic])

        for i, (nl, nh) in enumerate(zip(index[:,0], index[:,1])):
                M = ds[exp]['flow']['N'][nl:nh, idf]
                for n in range(Nharmonic):
                        Q = ds[exp]['flow']['Qn'][nl:nh, idf, n]
                        obs[i,n], obs_err[i,n] = obs_and_err(Q, M)
        return {'Name': 'vn', 'cenM': cenM, 'pTM' : None,
                        'obs': obs, 'err': obs_err}

def calculate_diff_vn(ds, exp, cenbins, pTbins, idf, pid='chg'):
        Ne = len(ds)
        pTbins = np.array(pTbins)
        cenbins = np.array(cenbins)
        cenM = np.mean(cenbins, axis=1)
        pTM = np.mean(pTbins, axis=1)
        Cindex = (cenbins/100.*Ne).astype(int)

        obs = 'd_flow_pid'
        data = ds[obs][:,idf][pid]

        # need soft flow within the same centrality bin first
        # only needs Ncen x [v2, v3]
        vnref = calculate_vn(ds, exp, cenbins, idf)

        # calculate hard vn
        vn = np.zeros([len(cenM), len(pTM), Nharmonic_diff])
        vn_err = np.zeros([len(cenM), len(pTM), Nharmonic_diff])
        for i, (nl, nh) in enumerate(Cindex):
                for j, (pl, ph) in enumerate(pTbins):
                        for n in range(Nharmonic_diff):
                                #w = data['N'][nl:nh, j] * ds[exp]['flow']['N'][nl:nh, idf]
                                w = (data['N'][nl:nh, j]).astype(float) * (ds[exp]['flow']['N'][nl:nh, idf]).astype(float) + 1e-9
                                dn2 = (data['Qn'][nl:nh,j,n].conjugate() * ds[exp]['flow']['Qn'][nl:nh, idf, n]).real / w
                                avg_dn2, std_avg_dn2 = weighted_mean_std(dn2, w)
                                vn[i, j, n] = avg_dn2/vnref['obs'][i,n]
                                vn_err[i, j, n] = std_avg_dn2/vnref['obs'][i,n]
        return {'Name': 'vn2', 'cenM': cenM, 'pTM' : pTM,
                        'obs': vn, 'err': vn_err}

def load_and_compute(inputfile, system, specify_idf=None):

    expt_type = expt_for_system[system]
    entry = np.zeros(1, dtype=np.dtype(bayes_dtype))
    res_unsort = np.fromfile(inputfile, dtype=return_result_dtype(expt_type))

    if specify_idf == None:
        idf_arr = [0, 1, 2, 3]
    else:
        idf_arr = [specify_idf]
    for idf in idf_arr:
        print("----------------------")
        print("idf : " + str(idf) )
        res = np.array(sorted(res_unsort, key=lambda x: x[expt_type][idf]['dNch_deta'], reverse=True))
        print("Result size : " + str(res.size))
        print("Number events w/o charged particles : " + str( (res_unsort[expt_type]['dNch_deta'][:, idf] == 0).sum() ) )

        # dNdeta
        tmp_obs='dNch_deta'
        try :
            cenb=np.array(obs_cent_list[system][tmp_obs])
            info = calculate_dNdeta(res, expt_type, cenb, idf)
            entry[system][tmp_obs]['mean'][:, idf] = info['obs']
            entry[system][tmp_obs]['err'][:,idf] = info['err']
        except KeyError :
            pass


        # dETdeta
        tmp_obs='dET_deta'
        try :
            cenb=np.array(obs_cent_list[system][tmp_obs])
            info = calculate_dETdeta(res, expt_type, cenb, idf)
            entry[system][tmp_obs]['mean'][:,idf] = info['obs']
            entry[system][tmp_obs]['err'][:,idf] = info['err']
        except KeyError :
            pass


        # dN(pid)/dy
        for s in ['pion', 'kaon', 'proton', 'Lambda', 'Omega', 'Xi', 'd']:
            try :
                cenb=np.array(obs_cent_list[system]['dN_dy_'+s])
                info = calculate_dNdy(res, expt_type, cenb, idf)
                entry[system]['dN_dy_'+s]['mean'][:,idf] = info['obs'][s]
                entry[system]['dN_dy_'+s]['err'][:,idf] = info['err'][s]
            except KeyError :
                pass


        # mean-pT
        for s in ['pion','kaon','proton', 'd']:
            try :
                cenb=np.array(obs_cent_list[system]['mean_pT_'+s])
                info = calculate_mean_pT(res, expt_type, cenb, idf)
                entry[system]['mean_pT_'+s]['mean'][:,idf] = info['obs'][s]
                entry[system]['mean_pT_'+s]['err'][:,idf] = info['err'][s]
            except KeyError:
                pass

        # mean-pT-fluct
        tmp_obs='pT_fluct'
        try :
            cenb=np.array(obs_cent_list[system][tmp_obs])
            info = calculate_mean_pT_fluct(res, expt_type, cenb, idf)
            entry[system][tmp_obs]['mean'][:,idf] = info['obs']
            entry[system][tmp_obs]['err'][:,idf] = info['err']
        except KeyError :
            pass

        # vn
        for n in range(2,5):
            tmp_obs='v'+str(n)+'2'
            try :
                cenb=np.array(obs_cent_list[system][tmp_obs])
                info = calculate_vn(res, expt_type, cenb, idf)
                entry[system][tmp_obs]['mean'][:,idf] = info['obs'][:, n-1]
                entry[system][tmp_obs]['err'][:,idf] = info['err'][:, n-1]
            except KeyError :
                pass

        # pid vn
        """
        pTbins = [[-10., 10.]]
        for n in range(2,3):
            for name, pid in Qn_species:
                print(name)
                tmp_obs='v'+str(n)+'2_' + name
                try :
                    cenb=np.array(obs_cent_list[system][tmp_obs])
                    info = calculate_diff_vn(res, expt_type, cenb, pTbins, idf, pid=name)
                    entry[system][tmp_obs]['mean'][:,idf] = info['obs'][:, n-1]
                    entry[system][tmp_obs]['err'][:,idf] = info['err'][:, n-1]
                except KeyError :
                    pass
        """

        from bins_and_cuts import ALICE_cent_bins

        # pT differential vn
        cenb = ALICE_cent_bins
        from calculations_file_format_single_event import Qn_diff_pT_cuts
        pTbins = []
        for i in range( len(Qn_diff_pT_cuts) - 1 ):
            pTbins.append( [Qn_diff_pT_cuts[i],  Qn_diff_pT_cuts[i+1]] )

        dir_str = 'model_calculations/MAP/' + idf_label_short[idf] + '/Predictions/diff_vn/'
        np.savetxt(dir_str + 'pT_bin_edges', Qn_diff_pT_cuts)

        for n in range(Nharmonic_diff):
            for name, pid in Qn_species:
                info = calculate_diff_vn(res, expt_type, cenb, pTbins, idf, pid=name)
                for icent, cent in enumerate(cenb):
                    cent_dir_str = str(cent[0]) + '-' + str(cent[1])
                    loc_dir_str = dir_str + cent_dir_str
                    file_str = loc_dir_str + '/' + name + '_v' + str(n+1)
                    obs_and_err = np.column_stack( (info['obs'][icent, :, n], info['err'][icent, :, n]) )
                    np.savetxt(file_str, obs_and_err)


        #pid dN/dpT
        for s,_ in Qn_species:
            cenb = ALICE_cent_bins
            info = calculate_dNdpT(res, expt_type, cenb, idf, s)

            # instead of saving dN/dpT to entry (changing bayes_dtype) ...
            # save it to its own file
            #file_str = 'diff_pT_spectra/idf_' + str(idf) + '/dN_dpT_' + s
            dir_str = 'model_calculations/MAP/' + idf_label_short[idf] + '/Predictions/diff_pT_spectra/'
            file_str = dir_str + 'dN_dpT_' + s
            np.savetxt(file_str, info['obs'])
            #save the pT bins to file
            np.savetxt(dir_str + 'pT_bin_edges', Qn_diff_pT_cuts)


    return entry

if __name__ == '__main__':

    system = system_strs[0]

    print("Computing observables for all design points")
    print("System = " + system)
    for folder_input, file_output, nset in zip(
              [f_events_main, f_events_validation],
              [f_obs_main, f_obs_validation],
              [n_design_pts_main, n_design_pts_validation],
           ):
        print("\n")
        print("Averaging events in " + folder_input)
        print("##########################")
        results = []
        for i in range(nset):
            print("design pt : " + str(i))
            filename = folder_input + "/{:d}.dat".format(i)
            results.append(load_and_compute(filename, system)[0])
            print("\n")
        results = np.array(results)
        print("results.shape = " + str(results.shape))
        results.tofile(file_output)
