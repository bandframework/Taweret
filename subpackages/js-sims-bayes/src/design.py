#!/usr/bin/env python3
"""
Generates Latin-hypercube parameter designs.

Writes input files for use with the JETSCAPE framework
Run ``python design.py --help`` for usage information.

.. warning::

    This module uses the R `lhs package
    <https://cran.r-project.org/package=lhs>`_ to generate maximin
    Latin-hypercube samples.  As far as I know, there is no equivalent library
    for Python (I am aware of `pyDOE <https://pythonhosted.org/pyDOE>`_, but
    that uses a much more rudimentary algorithm for maximin sampling).

    This means that R must be installed with the lhs package (run
    ``install.packages('lhs')`` in an R session).

"""

import itertools
import logging
from pathlib import Path
import subprocess
import os.path
import numpy as np

from configurations import *
from design_write_module_inputs import write_module_inputs

def generate_lhs(npoints, ndim, seed):
    """
    Generate a maximin Latin-hypercube sample (LHS) with the given number of
    points, dimensions, and random seed.

    """
    print(
        'generating maximin LHS: '
        'npoints = %d, ndim = %d, seed = %d',
        npoints, ndim, seed
    )

    print('generating using R')
    proc = subprocess.run(
        ['R', '--slave'],
        input="""
        library('lhs')
        set.seed({})
        write.table(maximinLHS({}, {}), col.names=FALSE, row.names=FALSE)
        """.format(seed, npoints, ndim).encode(),
        stdout=subprocess.PIPE,
        check=True
    )

    lhs = np.array(
        [l.split() for l in proc.stdout.splitlines()],
        dtype=float
    )

    return lhs


class Design:
    """
    Latin-hypercube model design.

    Creates a design for the given system with the given number of points.
    Creates the main (training) design if `validation` is false (default);
    creates the validation design if `validation` is true.  If `seed` is not
    given, a default random seed is used (different defaults for the main and
    validation designs).

    Public attributes:

    - ``system``: the system string
    - ``projectiles``, ``beam_energy``: system projectile pair and beam energy
    - ``type``: 'main' or 'validation'
    - ``keys``: list of parameter keys
    - ``labels``: list of parameter display labels (for TeX / matplotlib)
    - ``range``: list of parameter (min, max) tuples
    - ``min``, ``max``: numpy arrays of parameter min and max
    - ``ndim``: number of parameters (i.e. dimensions)
    - ``points``: list of design point names (formatted numbers)
    - ``array``: the actual design array

    The class also implicitly converts to a numpy array.

    This is probably the worst class in this project, and certainly the least
    generic.  It will probably need to be heavily edited for use in any other
    project, if not completely rewritten.

    """
    def __init__(self, system, validation, seed=None):
        n_design_pts_main = SystemsInfo[system[0]+"-"+system[1]+"-"+str(system[2])]['n_design']
        n_design_pts_validation = SystemsInfo[system[0]+"-"+system[1]+"-"+str(system[2])]['n_validation']
        npoints = n_design_pts_validation if validation else n_design_pts_main
        self.system = system[0]+system[1]+"-"+str(system[2])
        self.projectiles, self.target, self.beam_energy = system
        self.type = 'validation' if validation else 'main'
        print("system = ", self.system)

        # 5.02 TeV has ~1.2x particle production as 2.76 TeV
        # [https://inspirehep.net/record/1410589]
        norm_range = {
             200: (4., 9.),
            2760: (10., 20.),
            5020: (15., 25.),
            5440: {15., 25.}
        }[self.beam_energy]

        #any keys which are uncommented will be sampled / part of the design matrix
        self.keys, self.labels, self.range = map(list, zip(*[

        #trento
        ('norm', r'$N$[${:1.2f}$TeV]'.format(self.beam_energy/1000), (norm_range)),
        ('trento_p', r'$p$',                  ( -0.7,   0.7 )),
        ('sigma_k', r'$\sigma_k$',            ( 0.3,    2.0 )),
        ('nucleon_width', r'$w$ [fm]',        ( 0.5,    1.5 )),
        ('dmin3', r'$d_{\mathrm{min}}^3$ [fm]', ( 0.0, 1.7**3 )),

        #freestreaming
        ('tau_R', r'$\tau_R$ [fm/$c$]', (  0.3, 2.0 )),
        ('alpha',  r'$\alpha$',         ( -0.3, 0.3 )),

        #shear visc
        ('eta_over_s_T_kink_in_GeV', r'$T_{\eta,\mathrm{kink}}$ [GeV]',              ( 0.13, 0.3 )),
        ('eta_over_s_low_T_slope_in_GeV', r'$a_{\eta,\mathrm{low}}$ [GeV${}^{-1}$]', ( -2.0, 1.0 )),
        ('eta_over_s_high_T_slope_in_GeV',r'$a_{\eta,\mathrm{high}}$ [GeV${}^{-1}$]',( -1.0, 2.0 )),
        ('eta_over_s_at_kink', r'$(\eta/s)_{\mathrm{kink}}$',                        ( .01, 0.2 )),

        #bulk visc
        ('zeta_over_s_max',             r'$(\zeta/s)_{\max}$' , (0.01,  0.2)),
        ('zeta_over_s_T_peak_in_GeV',   r'$T_{\zeta,c}$ [GeV]', (0.12,  0.3)),
        ('zeta_over_s_width_in_GeV',   r'$w_{\zeta}$ [GeV]'   , (0.025 ,  0.15)),
        ('zeta_over_s_lambda_asymm',    r'$\lambda_{\zeta}$'  , (-0.8,  0.8)),

        #relaxation times
        ('shear_relax_time_factor',  r'$b_{\pi}$' ,  ( 2.0, 8.0 )),
        #('bulk_relax_time_factor',   r'$b_{\Pi}$' , ( 1.5, 10. )),
        #('bulk_relax_time_power',   r'$q_{\Pi}$' ,  ( 0.0, 2.5 )),

        #particlization temp
        ('Tswitch',  r'$T_{\mathrm{sw}}$ [GeV]', (0.13,  0.165)),
        ]))

        self.ndim = len(self.range)
        self.min, self.max = map(np.array, zip(*self.range))
        self.points = [str(i) for i in range(npoints)]

        #define different seeds for each system
        main_seeds = {
                    'AuAu-200' : 450829120,
                    'PbPb-2760': 450829121,
                    'PbPb-5020': 450829122,
                    'XeXe-5440': 450829123,
                    }
        validation_seeds = {
                    'AuAu-200' : 751783496,
                    'PbPb-2760': 751783497,
                    'PbPb-5020': 751783498,
                    'XeXe-5440': 751783499,
                    }

        #The seed is fixed here, which fixes the design points
        #if seed is None:
        #    seed = 751783496 if validation else 450829120

        #this fixes a different seed for each ystem s.t. each system samples
        #different points in the parameter space
        if validation :
            seed = validation_seeds[self.system]
        else :
            seed = main_seeds[self.system]

        self.array = self.min + (self.max - self.min)*generate_lhs(npoints=npoints, ndim=self.ndim, seed=seed)

    def __array__(self):
        return self.array

    def write_files(self, basedir):
        """
        Write input files for each design point to `basedir`.

        """

        # Directory where the input files will be saved
        outdir = basedir / self.type / self.system
        outdir.mkdir(parents=True, exist_ok=True)

        # File where a summary of the design points will be saved
        with open(os.path.join(basedir, 'design_points_'+str(self.type)\
                   +'_'+str(self.system)+'.dat'), 'w') as f:
            #write header
            f.write("idx")
            for key in self.keys:
                f.write(","+key)
            f.write("\n")
            for point, row in zip(self.points, self.array):
                f.write(str(point))
                for item in row:
                    f.write(",{:1.5f}".format(item))
                f.write("\n")
        # write parameter ranges to file to be imported by emulator
        with open(os.path.join(basedir, 'design_ranges_'+str(self.type)\
                    +'_'+str(self.system)+'.dat'), 'w') as f:
            # write header
            f.write("param,min,max\n")
            for key, minmax in zip(self.keys, self.range):
                f.write('{:s},{:1.5f},{:1.5f}\n'.format(key, *minmax))
        # write latex labels
        with open(os.path.join(basedir, 'design_labels_' + str(self.system) + '.dat'), 'w') as f:
            for item in self.labels:
                f.write(item + "\n")


        # Write the module input files for JETSCAPE-SIMS
        for point, row in zip(self.points, self.array):
            kwargs = dict(
                zip(self.keys, row),
            )
            write_module_inputs(
                outdir = str(outdir),
                design_point_id = point,
                #trento
                projectile = self.projectiles,
                target = self.projectiles,
                sqrts = self.beam_energy,
                inel_nucleon_cross_section = { # sqrt(s) [GeV] : sigma_NN [fm^2]
                                            200: 4.3,  2760: 6.4,  5020: 7.0, 5440: 7.1
                                             }[self.beam_energy],
                trento_normalization = kwargs['norm'],
                trento_reduced_thickness = kwargs['trento_p'],
                trento_fluctuation_k = 1.0 / kwargs['sigma_k']**2.0,
                trento_nucleon_width = kwargs['nucleon_width'],
                trento_nucleon_min_dist  = kwargs['dmin3']**(1.0/3.0),
                #freestreaming
                tau_R = kwargs['tau_R'],
                alpha = kwargs['alpha'],
                #shear
                eta_over_s_T_kink_in_GeV =       kwargs['eta_over_s_T_kink_in_GeV'],
                eta_over_s_low_T_slope_in_GeV =  kwargs['eta_over_s_low_T_slope_in_GeV'],
                eta_over_s_high_T_slope_in_GeV = kwargs['eta_over_s_high_T_slope_in_GeV'],
                eta_over_s_at_kink =             kwargs['eta_over_s_at_kink'],
                #bulk
                zeta_over_s_max =           kwargs['zeta_over_s_max'],
                zeta_over_s_width_in_GeV =  kwargs['zeta_over_s_width_in_GeV'],
                zeta_over_s_T_peak_in_GeV = kwargs['zeta_over_s_T_peak_in_GeV'],
                zeta_over_s_lambda_asymm =  kwargs['zeta_over_s_lambda_asymm'],
                #relaxation times
                shear_relax_time_factor = kwargs['shear_relax_time_factor'],
                #bulk_relax_time_factor = kwargs['bulk_relax_time_factor'],
                #bulk_relax_time_power = kwargs['bulk_relax_time_power'],
                #particlization
                T_switch = kwargs['Tswitch'],
            )

def main():
    import argparse
    parser = argparse.ArgumentParser(description='generate design input files')
    parser.add_argument('inputs_dir', type=Path, help='directory to place input files')
    args = parser.parse_args()

    for system, validation in itertools.product(systems, [False, True]):
        Design(system, validation=validation).write_files(args.inputs_dir)
    print('wrote all files to %s', args.inputs_dir)

if __name__ == '__main__':
    main()
