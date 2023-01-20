#!/usr/bin/env python3
"""
Markov chain Monte Carlo model calibration using the `affine-invariant ensemble
sampler (emcee) <http://dfm.io/emcee>`_.

This module must be run explicitly to create the posterior distribution.
Run ``python -m src.mcmc --help`` for complete usage information.

On first run, the number of walkers and burn-in steps must be specified, e.g.
::

    python -m src.mcmc --nwalkers 500 --nburnsteps 100 200

would run 500 walkers for 100 burn-in steps followed by 200 production steps.
This will create the HDF5 file :file:`mcmc/chain.hdf` (default path).

On subsequent runs, the chain resumes from the last point and the number of
walkers is inferred from the chain, so only the number of production steps is
required, e.g. ::

    python -m src.mcmc 300

would run an additional 300 production steps (total of 500).

To restart the chain, delete (or rename) the chain HDF5 file.
"""

import argparse
from contextlib import contextmanager
import logging
logging.getLogger().setLevel(logging.INFO)
import pandas as pd
import time

import emcee
import ptemcee
import h5py
import numpy as np
from scipy.linalg import lapack
from multiprocessing import Pool
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
from configurations import *
from emulator import Trained_Emulators, Trained_Emulators_all_df, _Covariance
from bayes_exp import Y_exp_data

#define groups of observables which are assumed to have correlated experimental error
expt_obs_corr_group = {
                        'dNch_deta' : 'yields',
                        'dET_deta' : 'yields',
                        'dN_dy_pion' : 'yields',
                        'dN_dy_kaon' : 'yields',
                        'dN_dy_proton' : 'yields',
                        'mean_pT_pion' : 'mean_pT',
                        'mean_pT_kaon' : 'mean_pT',
                        'mean_pT_proton' : 'mean_pT',
                        'pT_fluct' : 'pT_fluct',
                        'v22' : 'flows',
                        'v32' : 'flows',
                        'v42' : 'flows'
                        }

def mvn_loglike(y, cov):
    """
    Evaluate the multivariate-normal log-likelihood for difference vector `y`
    and covariance matrix `cov`:

        log_p = -1/2*[(y^T).(C^-1).y + log(det(C))] + const.

    This likelihood is NOT NORMALIZED, since this does not affect parameter estimation.
    The normalization const = -n/2*log(2*pi), where n is the dimensionality.

    Arguments `y` and `cov` MUST be np.arrays with dtype == float64 and shapes
    (n) and (n, n), respectively.  These requirements are NOT CHECKED.

    The calculation follows algorithm 2.1 in Rasmussen and Williams (Gaussian
    Processes for Machine Learning).

    """

    # Compute the Cholesky decomposition of the covariance.
    # Use bare LAPACK function to avoid scipy.linalg wrapper overhead.
    L, info = lapack.dpotrf(cov, clean=False)

    if info < 0:
        raise ValueError(
            'lapack dpotrf error: '
            'the {}-th argument had an illegal value'.format(-info)
        )
    elif info < 0:
        raise np.linalg.LinAlgError(
            'lapack dpotrf error: '
            'the leading minor of order {} is not positive definite'
            .format(info)
        )

    # Solve for alpha = cov^-1.y using the Cholesky decomp.
    alpha, info = lapack.dpotrs(L, y)

    if info != 0:
        raise ValueError(
            'lapack dpotrs error: '
            'the {}-th argument had an illegal value'.format(-info)
        )

    return -.5*np.dot(y, alpha) - np.log(L.diagonal()).sum()

def normed_mvn_loglike(y, cov):
    """
    Evaluate the multivariate-normal log-likelihood for difference vector `y`
    and covariance matrix `cov`:

        log_p = -1/2*[(y^T).(C^-1).y + log(det(C))] + const.

    This likelihood IS NORMALIZED.
    The normalization const = -n/2*log(2*pi), where n is the dimensionality.

    Arguments `y` and `cov` MUST be np.arrays with dtype == float64 and shapes
    (n) and (n, n), respectively.  These requirements are NOT CHECKED.

    The calculation follows algorithm 2.1 in Rasmussen and Williams (Gaussian
    Processes for Machine Learning).

    """

    # Compute the Cholesky decomposition of the covariance.
    # Use bare LAPACK function to avoid scipy.linalg wrapper overhead.
    L, info = lapack.dpotrf(cov, clean=False)

    if info < 0:
        raise ValueError(
            'lapack dpotrf error: '
            'the {}-th argument had an illegal value'.format(-info)
        )
    elif info < 0:
        raise np.linalg.LinAlgError(
            'lapack dpotrf error: '
            'the leading minor of order {} is not positive definite'
            .format(info)
        )

    # Solve for alpha = cov^-1.y using the Cholesky decomp.
    alpha, info = lapack.dpotrs(L, y)

    if info != 0:
        raise ValueError(
            'lapack dpotrs error: '
            'the {}-th argument had an illegal value'.format(-info)
        )


    n = len(y)
    norm_const = -n / ( 2. * np.log(2. * np.pi) )
    return -.5*np.dot(y, alpha) - np.log(L.diagonal()).sum() + norm_const


class LoggingEnsembleSampler(emcee.EnsembleSampler):
    def run_mcmc(self, X0, nsteps, status=None, **kwargs):
        """
        Run MCMC with logging every 'status' steps (default: approx 10% of
        nsteps).

        """

        print('running {:d} walkers for {:d} steps'.format(self.k, nsteps))
        if status is None:
            status = nsteps // 10

        for n, result in enumerate( self.sample(X0, iterations=nsteps, **kwargs), start=1 ):
            if n % status == 0 or n == nsteps:
                af = self.acceptance_fraction
                print(
                    'step {:d}: acceptance fraction: '
                    'mean {:.4f}, std {:.4f}, min {:.4f}, max {:.4f}'.format(
                    n, af.mean(), af.std(), af.min(), af.max()
                    )
                    )

        return result

def compute_cov(system, obs1, obs2, dy1, dy2):
    x1 = np.linspace(0, 1, len(dy1))
    x2 = np.linspace(0, 1, len(dy2))
    if assume_corr_exp_error:
        if obs1 == obs2:
            cov_mat = np.outer(dy1, dy2) * np.exp(-np.subtract.outer(x1, x2)**2 / cent_corr_length**2)
        elif expt_obs_corr_group[obs1] == expt_obs_corr_group[obs2]:
            cov_mat = np.outer(dy1, dy2) * np.exp(-np.subtract.outer(x1, x2)**2 / cent_corr_length**2) * 0.8
        else:
            cov_mat = np.zeros([dy1.size, dy2.size])

    else:
        if obs1 == obs2:
            cov_mat = np.diag(dy1**2)
        else:
            cov_mat = np.zeros([dy1.size, dy2.size])

    return cov_mat

class Chain:
    """
    High-level interface for running MCMC calibration and accessing results.

    Currently all design parameters except for the normalizations are required
    to be the same at all beam energies.  It is assumed (NOT checked) that all
    system designs have the same parameters and ranges (except for the norms).

    """
    def __init__(self, path=workdir/'mcmc'/'chain-idf-{:d}.hdf'.format(idf)):
        self.path = path
        self.path.parent.mkdir(exist_ok=True)

        self._slices = {}
        self._expt_y = {}
        self._expt_cov = {}

        Yexp = Y_exp_data

        # For multi-system calibration, we need to specify what parameters
        # are considered universal (the same across all system), and what
        # parameters are allowed to be a function of projectile / target /
        # collision energy
        # Now, for simplicity, I only let the energy density normalization
        # of the initial condition to be system-dependent, while others are
        # universal in the high-energy limit
        set_common = False
        self.sysdep_max = []
        self.sysdep_min = []
        self.sysdep_labels = []
        self.sys_idx = {} # keep track what parameters are used for each system
        Nsys = len(system_strs)
        for i, s in enumerate(system_strs):
            design, design_min, design_max, labels = load_design(s, pset='main')
            self.sysdep_max.append(design_max[0])
            self.sysdep_min.append(design_min[0])
            self.sysdep_labels.append(labels[0])
            self.sys_idx[s] = [i]+list(range(Nsys, Nsys+len(design_max)-1))
            if not set_common:
                #right now the extra std. deviation is hardcoded in, want to fix/generalize this in the future!
                self.common_max = np.array(list(design_max[1:])+[.4]) # the last entry is max value for the extra std. deviation
                self.common_min = np.array(list(design_min[1:])+[1e-3]) # the last value is the min. for the extra std. deviation
                self.common_labels = list(labels[1:]) + [r'$\sigma_M$']
                self.common_ndim = len(self.common_max)
                self.common_range = np.array([self.common_min,
                                                self.common_max]).T
        self.sysdep_range = np.array([self.sysdep_min, self.sysdep_max]).T
        self.sysdep_ndim = len(self.sysdep_max)
        self.max = np.concatenate([self.sysdep_max, self.common_max])
        self.min = np.concatenate([self.sysdep_min, self.common_min])
        self.range = np.array([self.min, self.max]).T
        self.labels =  self.sysdep_labels + self.common_labels
        self.ndim = self.sysdep_ndim + self.common_ndim

        if hold_parameters:
            self.hold = hold_parameters_set
            # modify the range of the holding parameters to a delta-
            # function like range:
            #       p0-delta < p < p0+delta,
            # this range is not used in MCMC but only for plotting
            # the hold values of p0 will be passed to emulator directly
            for (idx, value) in self.hold:
                if value < self.min[idx] or self.max[idx] < value:
                    print("hold value out of range for idx = ", idx, ", value = ", value)
                    print("range is [" + str(self.min[idx]) + " , " + str(self.max[idx]) + " ]")
                    exit(-1)
                delta = (self.max[idx]-self.min[idx])/1e3

                self.min[idx]= value-delta
                self.max[idx]= value+delta

        if change_parameters_range:
            self.change_range = change_parameters_range_set
            # modify the ranges of a set of parameters
            for (idx, pmin, pmax) in self.change_range:
                if pmin < self.min[idx] or self.max[idx] < pmax:
                    print("update range value out of design range for idx = " + str(idx) + " : pmin = " + str(pmin) + " , pmax = " + str(pmax) )
                    print("design range is [" + str(self.min[idx]) + " , " + str(self.max[idx]) + " ]")
                    exit(-1)

                self.min[idx]= pmin
                self.max[idx]= pmax

        #the volume of the uniform prior
        diff =  self.max - self.min
        self.prior_volume = np.prod( diff )

        print("Pre-compute experimental covariance matrix")
        if change_exp_error:
            print("WARNING! Multiplying experimental error by values in change_exp_error_vals : ")
            print(change_exp_error_vals)
            for s in system_strs:
                for obs in change_exp_error_vals[s].keys():
                    Yexp[s][obs]['err'] *= change_exp_error_vals[s][obs]

        for s in system_strs:
            nobs = 0
            self._slices[s] = []

            for obs in active_obs_list[s]:
                try:
                    obsdata = Yexp[s][obs]['mean'][idf]
                except KeyError:
                    print("KeyError in building slices")
                    continue

                n = obsdata.size
                self._slices[s].append(
                        (obs, slice(nobs, nobs + n))
                  )
                nobs += n

            self._expt_y[s] = np.empty(nobs)
            self._expt_cov[s] = np.empty((nobs, nobs))

            for obs1, slc1 in self._slices[s]:
                is_mult_1 = ('dN' in obs1) or ('dET' in obs1)
                if is_mult_1 and transform_multiplicities:
                    self._expt_y[s][slc1] = np.log(Yexp[s][obs1]['mean'][idf] + 1.)
                    dy1 = Yexp[s][obs1]['err'][idf] / (Yexp[s][obs1]['mean'][idf] + 1.)
                else :
                    self._expt_y[s][slc1] = Yexp[s][obs1]['mean'][idf]
                    dy1 = Yexp[s][obs1]['err'][idf]

                for obs2, slc2 in self._slices[s]:
                    is_mult_2 = ('dN' in obs2) or ('dET' in obs2)
                    if is_mult_2 and transform_multiplicities:
                        dy2 = Yexp[s][obs2]['err'][idf] / (Yexp[s][obs2]['mean'][idf] + 1.)
                    else :
                        dy2 = Yexp[s][obs2]['err'][idf]

                    self._expt_cov[s][slc1, slc2] = compute_cov(s, obs1, obs2, dy1, dy2)


    def _predict(self, X, **kwargs):
        """
        Call each system emulator to predict model output at X. (using df model specified by idf in configurations.py)

        """
        if hold_parameters:
            for (idx, value) in self.hold:
                X[:,idx] = value
        return { s: Trained_Emulators[s].predict(X[:,self.sys_idx[s]], **kwargs) for s in system_strs }

    def _predict_given_df(self, X, idf, **kwargs):
        """
        Call each system emulator for a given idf model (argument) to predict model output at X.

        """
        if hold_parameters:
            for (idx, value) in self.hold:
                X[:,idx] = value
        return { s: Trained_Emulators_all_df[s][idf].predict(X[:,self.sys_idx[s]], **kwargs) for s in system_strs }


    def log_posterior(self, X, extra_std_prior_scale=0.001):
        """
        Evaluate the posterior at `X`.

        `extra_std_prior_scale` is the scale parameter for the prior
        distribution on the model sys error parameter:

            prior ~ sigma^2 * exp(-sigma/scale)

        """
        X = np.array(X, copy=False, ndmin=2)
        lp = np.zeros(X.shape[0])
        inside = np.all((X > self.min) & (X < self.max), axis=1)
        lp[~inside] = -np.inf
        extra_std = X[inside, -1]

        nsamples = np.count_nonzero(inside)
        if nsamples > 0:
            pred = self._predict( X[inside], return_cov=True, extra_std=extra_std )
            for sys in system_strs:
                nobs = self._expt_y[sys].size
                # allocate difference (model - expt) and covariance arrays
                dY = np.empty((nsamples, nobs))
                cov = np.empty((nsamples, nobs, nobs))

                Y_pred, cov_pred = pred[sys]

                # copy predictive mean and covariance into allocated arrays
                for obs1, slc1 in self._slices[sys]:
                    dY[:, slc1] = Y_pred[obs1] - self._expt_y[sys][slc1]
                    for obs2, slc2 in self._slices[sys]:
                        cov[:, slc1, slc2] = cov_pred[obs1, obs2]

                # add expt cov to model cov
                cov += self._expt_cov[sys]

                # compute log likelihood at each point, w/o normalization
                lp[inside] += list(map(mvn_loglike, dY, cov))

            # add prior for extra_std (model sys error)
            lp[inside] += 2*np.log(extra_std) - extra_std/extra_std_prior_scale


        return lp

    def log_prior(self, X):
        """
        Evaluate the prior at `X`.

        """
        X = np.array(X, copy=False, ndmin=2)

        #not normalized
        #lp = np.zeros(X.shape[0])

        #normalize the prior
        lp = np.log( np.ones(X.shape[0]) / self.prior_volume )

        inside = np.all((X > self.min) & (X < self.max), axis=1)
        lp[~inside] = -np.inf

        return lp

    def log_likelihood(self, X, extra_std_prior_scale=0.001):
        """
        Evaluate the likelihood at `X`.

        """
        X = np.array(X, copy=False, ndmin=2)

        lp = np.zeros(X.shape[0])

        inside = np.all( (X > self.min) & (X < self.max), axis=1)
        lp[~inside] = -np.inf

        extra_std = X[inside, -1]

        nsamples = np.count_nonzero(inside)
        if nsamples > 0:
            pred = self._predict( X[inside], return_cov=True, extra_std=extra_std )
            for sys in system_strs:
                nobs = self._expt_y[sys].size
                # allocate difference (model - expt) and covariance arrays
                dY = np.empty((nsamples, nobs))
                cov = np.empty((nsamples, nobs, nobs))

                Y_pred, cov_pred = pred[sys]

                # copy predictive mean and covariance into allocated arrays
                for obs1, slc1 in self._slices[sys]:
                    dY[:, slc1] = Y_pred[obs1] - self._expt_y[sys][slc1]
                    for obs2, slc2 in self._slices[sys]:
                        cov[:, slc1, slc2] = cov_pred[obs1, obs2]

                #TEMPORARY FIX EMULATION COVARIANCE TO ZERO
                #cov = np.empty((nsamples, nobs, nobs))

                #TEMPORARY REDUCE EMU COVARIANCE BY FACTOR
                #cov *= 0.1

                # add expt cov to model cov
                cov += self._expt_cov[sys]

                # compute normalized log likelihood at each point
                lp[inside] += list(map(normed_mvn_loglike, dY, cov))

            # add prior for extra_std (model sys error)
            lp[inside] += 2*np.log(extra_std) - extra_std/extra_std_prior_scale


        return lp


    def random_pos(self, n=1):
        """
        Generate `n` random positions in parameter space.

        """
        return np.random.uniform(self.min, self.max, (n, self.ndim))

    @staticmethod
    def map(f, args):
        """
        Dummy function so that this object can be used as a 'pool' for
        :meth:`emcee.EnsembleSampler`.

        """
        return f(args)

    def run_mcmc(self, nsteps, nburnsteps=None, nwalkers=None, status=None, ntemps=1):
        """
        Run MCMC model calibration.  If the chain already exists, continue from
        the last point, otherwise burn-in and start the chain.

        """
        with self.open('a') as f:
            try:
                dset = f['chain']
            except KeyError:
                burn = True
                if nburnsteps is None or nwalkers is None:
                    print('must specify nburnsteps and nwalkers to start chain')
                    return
                dset = f.create_dataset(
                    'chain', dtype='f8',
                    shape=(nwalkers, 0, self.ndim),
                    chunks=(nwalkers, 1, self.ndim),
                    maxshape=(nwalkers, None, self.ndim),
                    compression='lzf'
                )
            else:
                burn = False
                nwalkers = dset.shape[0]

            #choose number of temperatures for PTSampler
            if usePTSampler:
                print("Using PTSampler")
                print("ntemps = " + str(ntemps))
                ncpu = cpu_count()
                print("{0} CPUs".format(ncpu))
                Tmax=np.inf
                with Pool() as pool:
                    sampler=ptemcee.Sampler(nwalkers, self.ndim, self.log_likelihood, self.log_prior, ntemps, Tmax, pool=pool)
                    print("Running burn-in phase")
                    nburn0 = nburnsteps
                    pos0 = np.random.uniform(self.min, self.max, (ntemps, nwalkers, self.ndim))
                    start = time.time()
                    sampler.run_mcmc(pos0, nburn0, adapt=True)
                    end = time.time()
                    print("... finished in " + str(end - start) + " sec")
                    print("sampler.chain.shape " + str(sampler.chain.shape))
                    print("betas = " + str(sampler.betas))
                    #get the last step of the chain
                    pos0 = sampler.chain[:, :, -1, :]
                    print("pos0.shape " + str(pos0.shape))
                    sampler.reset()
                    print("Running MCMC chains")
                    niters = 10
                    for iter in range(niters):
                        print("betas = " + str(sampler.betas))
                        print("iteration " + str(iter) + " ...")
                        start = time.time()
                        sampler.run_mcmc(pos0, nsteps // 10)
                        end = time.time()
                        logZ, dlogZ = sampler.log_evidence_estimate()
                        print("logZ = " + str(logZ) + " +/- " + str(dlogZ))
                        print("... finished in " + str(end - start) + " sec")


                print("sampler.chain.shape " + str(sampler.chain.shape))
                print('writing chain to file')
                dset.resize(dset.shape[1] + nsteps, 1)
                #save only the zero temperature chain
                dset[:, -nsteps:, :] = sampler.chain[0, :, :, :]

                #save the thermodynamic log evidence
                #logZ, dlogZ = sampler.thermodynamic_integration_log_evidence()
                logZ, dlogZ = sampler.log_evidence_estimate()
                print("logZ = " + str(logZ) + " +/- " + str(dlogZ))
                with open('mcmc/chain-idf-' + str(idf) + '-info.dat', 'w') as f:
                    f.write('logZ ' + str(logZ) + '\n')
                    f.write('dlogZ ' + str(dlogZ))


            else:
                sampler = LoggingEnsembleSampler(nwalkers, self.ndim, self.log_posterior, pool=self)
                if burn:
                    print('no existing chain found, starting initial burn-in')
                    # Run first half of burn-in starting from random positions.
                    nburn0 = nburnsteps // 2
                    sampler.run_mcmc( self.random_pos(nwalkers), nburn0, status=status )
                    print('resampling walker positions')
                    # Reposition walkers to the most likely points in the chain,
                    # then run the second half of burn-in.  This significantly
                    # accelerates burn-in and helps prevent stuck walkers.
                    X0 = sampler.flatchain[ np.unique( sampler.flatlnprobability, return_index=True )[1][-nwalkers:] ]
                    sampler.reset()
                    X0 = sampler.run_mcmc(X0, nburnsteps - nburn0, status=status, storechain=False)[0]
                    sampler.reset()
                    print('burn-in complete, starting production')
                else:
                    print('restarting from last point of existing chain')
                    X0 = dset[:, -1, :]
                sampler.run_mcmc(X0, nsteps, status=status)
                print('writing chain to file')
                dset.resize(dset.shape[1] + nsteps, 1)
                dset[:, -nsteps:, :] = sampler.chain

    def open(self, mode='r'):
        """
        Return a handle to the chain HDF5 file.

        """
        return h5py.File(str(self.path), mode)

    @contextmanager
    def dataset(self, mode='r', name='chain'):
        """
        Context manager for quickly accessing a dataset in the chain HDF5 file.

        >>> with Chain().dataset() as dset:
                # do something with dset object

        """
        with self.open(mode) as f:
            yield f[name]

    def load(self, thin=1):
        """
        Read the chain from file.  If `keys` are given, read only those
        parameters.  Read only every `thin`'th sample from the chain.

        """
        ndim = self.ndim
        indices = slice(None)

        with self.dataset() as d:
            return np.array(d[:, ::thin, indices]).reshape(-1, ndim)

    def load_wo_reshape(self, thin=1):
        """
        Read the chain from file.  If `keys` are given, read only those
        parameters.  Read only every `thin`'th sample from the chain. Don't reshape chain

        """
        #ndim = self.ndim
        indices = slice(None)

        with self.dataset() as d:
            #return np.array(d[:, ::thin, indices]).reshape(-1, ndim)
            return np.array(d[:, ::thin, indices])

    def samples(self, n=1):
        """
        Predict model output at `n` parameter points randomly drawn from the
        chain. (Uses emulator given by idf setting in configurations.py)

        """
        with self.dataset() as d:
            X = np.array([
                d[i] for i in zip(*[
                    np.random.randint(s, size=n) for s in d.shape[:2]
                ])
            ])

        return self._predict(X)

    def samples_given_df(self, idf, n=1):
        """
        Predict model output at `n` parameter points randomly drawn from the
        chain using a specific df model emulator selected by input argument.

        """
        with self.dataset() as d:
            X = np.array([
                d[i] for i in zip(*[
                    np.random.randint(s, size=n) for s in d.shape[:2]
                ])
            ])

        return self._predict_given_df(X, idf)


def credible_interval(samples, ci=.9):
    """
    Compute the highest-posterior density (HPD) credible interval (default 90%)
    for an array of samples.

    """
    # number of intervals to compute
    nci = int((1 - ci)*samples.size)

    # find highest posterior density (HPD) credible interval
    # i.e. the one with minimum width
    argp = np.argpartition(samples, [nci, samples.size - nci])
    cil = np.sort(samples[argp[:nci]])   # interval lows
    cih = np.sort(samples[argp[-nci:]])  # interval highs
    ihpd = np.argmin(cih - cil)

    return cil[ihpd], cih[ihpd]


def main():
    parser = argparse.ArgumentParser(description='MCMC')

    parser.add_argument(
        'nsteps', type=int,
        help='number of steps'
    )
    parser.add_argument(
        '--nwalkers', type=int,
        help='number of walkers'
    )
    parser.add_argument(
        '--ntemps', type=int,
        help='number of points in temperature (for PTSampler only)'
    )
    parser.add_argument(
        '--nburnsteps', type=int,
        help='number of burn-in steps'
    )
    parser.add_argument(
        '--status', type=int,
        help='number of steps between logging status'
    )

    args = parser.parse_args()
    Chain().run_mcmc(
            nsteps=args.nsteps,
            nwalkers=args.nwalkers,
            nburnsteps=args.nburnsteps,
            status=args.status,
            ntemps=args.ntemps
          )


if __name__ == '__main__':
    main()
