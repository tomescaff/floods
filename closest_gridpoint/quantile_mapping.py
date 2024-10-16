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

def cdf_precip_drywet(x, dist_full, minval):
    """Computes the CDF of a value x in a precip. distribution dist_full.
    
    Params
    ------
    x: float
        Value to compute the CDF
    dist_full: np.array
        Full distribution
    minval: float
        Minimum value to consider in the distribution
        
    Returns
    -------
    float
        CDF of the value x
    """
    dist_wet = dist_full[dist_full > minval]
    dist_dry = dist_full[dist_full <= minval]
    cdf_minval = dist_dry.size/dist_full.size
    if x <= minval:
        cdf = dist_full[dist_full <= x].size/dist_full.size
    else:
        cdf_wet = gamma.cdf(x, *gamma.fit(dist_wet, floc=minval))
        cdf = cdf_minval + (1-cdf_minval)*cdf_wet
    return cdf

def invcdf_precip_drywet(p, dist_full, minval):
    """Inverse CDF of precipitation data.
    
    Params
    ------
    p: float
        Probability value
    dist_full: np.array
        Full distribution
    minval: float
        Minimum value to consider in the distribution
        
    Returns
    -------
    float
        inverse of CDF; x such that p = CDF(x)
    """
    cdf_minval = cdf_precip_drywet(minval, dist_full, minval)
    if p <= cdf_minval:
        return np.quantile(dist_full, p)
    else:
        dist_wet = dist_full[dist_full > minval]
        gamfit_obs = gamma.fit(dist_wet, floc=minval)
        z = gamma.ppf((p - cdf_minval)/(1-cdf_minval), *gamfit_obs)
        return z

cdf_fun_precip_drywet = np.vectorize(cdf_precip_drywet, excluded=[1, 2])
invcdf_fun_precip_drywet = np.vectorize(invcdf_precip_drywet, excluded=[1, 2])

def gen_transfer_fun_precip_drywet(observed_values, modeled_values, minval=3):
    """Generate transfer function for precipitation data, dry-wet approach.

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
    
    def transfer_fun_precip_drywet(z):
        p = cdf_fun_precip_drywet(z, modeled_values, minval)
        return invcdf_fun_precip_drywet(p, observed_values, minval)
    transfer_fun = np.vectorize(transfer_fun_precip_drywet)
    return transfer_fun

