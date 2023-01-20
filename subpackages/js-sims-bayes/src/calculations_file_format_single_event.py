# Data structure used to save hadronic observables
# (or quantities that can be used to compute hadronic observables)
# for each hydrodynamic event (oversamples or not)

from configurations import *

# species (name, ID) for identified particle observables
species = [
	('pion', 211),
	('kaon', 321),
	('proton', 2212),
	('Lambda', 3122),
	('Sigma0', 3212),
	('Xi', 3312),
	('Omega', 3334),
	('phi', 333),
	#('d', 1000010020),
]
pi_K_p = [
	('pion', 211),
	('kaon', 321),
	('proton', 2212),
]

Qn_species = [
        ('pion', 211),
        ('kaon', 321),
        ('proton', 2212),
        ('Sigma', 3222),
        ('Xi', 3312),
		#('d', 1000010020),
		('Ch', None)
]

#old
# Qn_diff_pT_cuts=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2.,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.,3.2,3.4,3.6,3.8,4.]
#updated fine
#Qn_diff_pT_cuts=[0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2.,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.,3.2,3.4,3.6,3.8,4.,10.]
#coarse
Qn_diff_pT_cuts=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

Qn_diff_NpT = len(Qn_diff_pT_cuts)-1
Nharmonic = 8
Nharmonic_diff = 5

# results "array" (one element)
# to be overwritten for each event
# four entries for four 'delta-f' options
number_of_viscous_corrections=4

def return_result_dtype(expt_type):
	result_dtype=[
	('initial_entropy', float_t, 1),
	('impact_parameter', float_t, 1),
	('npart', float_t, 1),
	(expt_type,
	        [
	                ('nsamples', int_t, 1),
	                # 1) dNch/deta, eta[-0.5, 0.5], charged
	                ('dNch_deta', float_t, 1),
	                # 2) dET/deta, eta[-0.6, 0.6]
	                ('dET_deta', float_t, 1),
	                # 3.1) The Tmunu observables, eta[-0.6, 0.6]
	                ('Tmunu', float_t, 10),
	                # 3.2) The Tmunu observables, eta[-0.5, 0.5], charged
	                ('Tmunu_chg', float_t, 10),
	                # 4.1) identified particle yield
	                ('dN_dy',       [(name, float_t, 1) for (name,_) in species], 1),
	                # 4.2) identified particle <pT>
	                ('mean_pT', [(name, float_t, 1) for (name,_) in species], 1),
	                # 5.1) pT fluct, pT[0.15, 2], eta[-0.8, 0.8], charged
	                ('pT_fluct_chg', [      ('N', int_t, 1),
	                                                        ('sum_pT', float_t, 1),
	                                                        ('sum_pT2', float_t, 1)], 1),
	                # 5.2) pT fluct, pT[0.15, 2], eta[-0.8, 0.8], pi, K, p
	                ('pT_fluct_pid', [      (name, [        ('N', int_t, 1),
	                                                                                ('sum_pT', float_t, 1),
	                                                                                ('sum_pT2', float_t, 1)], 1     )
	                                                          for (name,_) in pi_K_p        ], 1),
	                # 6) Q vector, pT[0.2, 5.0], eta [-0.8, 0.8], charged
	                ('flow', [      ('N', int_t, 1),
	                                        ('Qn', complex_t, Nharmonic)], 1),
	        ], number_of_viscous_corrections),
	# Q vector, diff-flow, identified charged hadrons
	('d_flow_pid', [(name, [('N', int_t, Qn_diff_NpT),
	                                                                ('Qn', complex_t, [Qn_diff_NpT, Nharmonic_diff])], 1)
	                                for (name,_) in Qn_species      ], number_of_viscous_corrections),
	]
	return result_dtype
