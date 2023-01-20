##!/usr/bin/env python3
#
import numpy as np
from collections.abc import Iterable
import h5py
import sys, os, glob

from configurations import *
from calculations_file_format_single_event import result_dtype, Qn_species, Qn_diff_pT_cuts


filename=sys.argv[1]
data = np.fromfile(filename, dtype=result_dtype)

system = 'Pb-Pb-2760'

Qn_rap_range=2.

mid_pT_bins=[(Qn_diff_pT_cuts[i]+Qn_diff_pT_cuts[i+1])/2. for i in range(0,len(Qn_diff_pT_cuts)-1)]

print(data['ALICE'].dtype)

Qn_diff=data['d_flow_pid']

#print(v1)
#
##print(Qn_diff['pion'])
##
##print(np.add(Qn_diff['pion'],Qn_diff['kaon']))
#


Qn_diff_ch=np.zeros(Qn_diff['pion'].shape,dtype=Qn_diff['pion'].dtype)
## Get charged hadron Q_n
#for species, pid in Qn_species:
#    weight=1
#    if (species== 'Sigma'):
#        weight=2
#    tmp_Qn_id=Qn_diff[species]
##    print(tmp_Qn_id['N'])
##    Qn_diff_ch['N']=Qn_diff_ch['N']+weight*tmp_Qn_id['N']
#    Qn_diff_ch['N']+=weight*tmp_Qn_id['N']
#    Qn_diff_ch['Qn']+=weight*tmp_Qn_id['Qn']

# Get charged hadron Q_n between pseudorapidity cuts

def integrated_jacobian(m, pT, etaCut):

    m2=m*m
    pT2=pT*pT
    cosh2EtaCut=np.cosh(2*etaCut)
    sinhEtaCut=np.sinh(etaCut)

    return np.log((np.sqrt(2*m2 + pT2 + pT2*cosh2EtaCut) + np.sqrt(2)*pT*sinhEtaCut)/(np.sqrt(2*m2 + pT2 + pT2*cosh2EtaCut) - np.sqrt(2)*pT*sinhEtaCut)) #/(2.*etaCut)

masses={
'pion':0.138,
'kaon':0.494,
'proton':0.938,
'Sigma':1.189,
'Xi':1.318
}

etaCut=0.8
pTminCut=0.2
#print(Qn_diff['pion']['N'][0][:][0])
#print(Qn_diff['pion']['N'][0][:,0])

Qn_ch=np.zeros(1,dtype=[('N', '<f8', 4), ('Qn', '<c16', (4,5))])

#print(Qn_ch['Qn'])
#
#print(Qn_diff['pion']['Qn'][0][:,0])
#
#exit(1)


#print(Qn_diff['pion']['N'][0][:][0].shape)
for species, pid in Qn_species:
    weight=1
    if (species== 'Sigma'):
        weight=2

    for i, pT in enumerate(mid_pT_bins):
        if (pT<pTminCut):
            continue
        rapidity_to_pseudorapidity_jacobian=integrated_jacobian(masses[species],pT,etaCut)
        Qn_ch['N']+=Qn_diff[species]['N'][0][:,i]/Qn_rap_range*weight*rapidity_to_pseudorapidity_jacobian
        Qn_ch['Qn']+=Qn_diff[species]['Qn'][0][:,i]/Qn_rap_range*weight*rapidity_to_pseudorapidity_jacobian
        #print(i,pT,Qn_diff[species]['Qn'][0,:,i])

#    tmp_Qn_id=Qn_diff[species]
##    print(tmp_Qn_id['N'])
##    Qn_diff_ch['N']=Qn_diff_ch['N']+weight*tmp_Qn_id['N']
#    Qn_diff_ch['N']+=weight*tmp_Qn_id['N']
#    Qn_diff_ch['Qn']+=weight*tmp_Qn_id['Qn']

print("Q0_ch",np.divide(data['ALICE']['flow']['N'],Qn_ch['N']))
#print("Qn_ch",data['ALICE']['flow']['Qn'][0,:,0:5],Qn_ch['Qn'])
print("Qn_ch",np.divide(data['ALICE']['flow']['Qn'][0,:,0:5],Qn_ch['Qn']))

for species, pid in Qn_species:
    alt=species
    if (species== 'Sigma'):
        alt='Sigma0'

    print('mult ',species,": ",data['ALICE']['dN_dy'][alt],np.divide(np.sum(Qn_diff[species]['N'],axis=2)/Qn_rap_range,data['ALICE']['nsamples']))
    print('mean pT ', species,": ",data['ALICE']['mean_pT'][alt],
    np.divide(np.sum(np.multiply(Qn_diff[species]['N'],mid_pT_bins),axis=2),np.sum(Qn_diff[species]['N'],axis=2)))
#    np.sum(np.multiply(Qn_diff[species]['N'],mid_pT_bins),axis=2)/np.sum(Qn_diff[species]['N'],axis=2) )
    #[np.average(mid_pT_bins, axis=2, weights=Qn_diff[species]['N'][i]) for i in range(0,4)])
#    print('test ', species, ": ", Qn_diff[species]['N'], mid_pT_bins)



## Loop over data structure
## Assumes that "data" is a numpy array with dtype given
## by the array "structure" (though the latter is not a dtype object)
#def print_data_structure(data, structure):
#
#    n_items=len(structure)
#
#    if (n_items > 0):
#        for n, item in enumerate(structure):
#            tmp_struct=structure[n]
#            # If the item has substructure, recurse on it
#            if (not isinstance(tmp_struct[1], str)) and (isinstance(tmp_struct[1], Iterable)):
#                print(tmp_struct[0])
#                print_data_structure(data[tmp_struct[0]],tmp_struct[1])
#            # If no substructure, just output the result
#            else:
#                print(tmp_struct[0],data[tmp_struct[0]])
#
#print_data_structure(data, result_dtype)
#
