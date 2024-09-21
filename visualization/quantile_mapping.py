"""Module for quantile mapping of modeled data."""

import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.stats import norm

def gen_transfer_fun_precip(observed_values, modeled_values, minval=3):
    """Generate transfer function for precipitation data.

    Parameters
    ----------
    observed_values : xarray.DataArray
        Observed precipitation data.
    modeled_values : xarray.DataArray
        Modeled precipitation data.
    minval : float, optional
        Lower threshold in mm. The default is 3.

    Returns
    -------
    transfer_fun : function
        Transfer function to correct modeled data.
    """
    
    obs_filt = observed_values.where(observed_values > minval, drop=True)
    mod_filt = modeled_values.where(modeled_values > minval, drop=True)
    
    gamfit_obs = gamma.fit(obs_filt, floc=minval)
    gamfit_mod = gamma.fit(mod_filt, floc=minval)
    
    def transfer_fun_precip(z):
        if z <= minval:
            return z
        else:
            return gamma.ppf(gamma.cdf(z, *gamfit_mod), *gamfit_obs)
    transfer_fun = np.vectorize(transfer_fun_precip)
    return transfer_fun

def gen_transfer_fun_H0(observed_values, modeled_values):
    """Generate transfer function for H0 data.

    Parameters
    ----------
    observed_values : xarray.DataArray
        Observed H0 data.
    modeled_values : xarray.DataArray
        Modeled H0 data.

    Returns
    -------
    transfer_fun : function
        Transfer function to correct modeled data.
    """
    
    normfit_obs = norm.fit(observed_values.values)
    normfit_mod = norm.fit(modeled_values.values)
    
    def transfer_fun_H0(z):
        return norm.ppf(norm.cdf(z, *normfit_mod), *normfit_obs)
    transfer_fun = np.vectorize(transfer_fun_H0)
    return transfer_fun

