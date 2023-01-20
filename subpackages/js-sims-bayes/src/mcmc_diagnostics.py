import numpy as np

def autocorrelation(chain, max_percent_lag):

    """A function to return the autocorrelation function
    for a time series. Useful for checking MCMC chain convergence
    and independent samples."""

    max_lag = int( np.floor(len(chain) * max_percent_lag) )
    dimension = len(chain)
    acors = np.empty(max_lag+1)
    #if max_lag > len(chain)/5:
    #    warnings.warn('max_lag is more than one fifth the chain length')
    # Create a copy of the chain with average zero
    chain1d = chain - np.average(chain)
    for lag in range(max_lag+1):
        unshifted = None
        shifted = chain1d[lag:]
        if 0 == lag:
            unshifted = chain1d
        else:
            unshifted = chain1d[:-lag]
        normalization = np.sqrt(np.dot(unshifted, unshifted))
        normalization *= np.sqrt(np.dot(shifted, shifted))
        acors[lag] = np.dot(unshifted, shifted) / normalization
    return acors
