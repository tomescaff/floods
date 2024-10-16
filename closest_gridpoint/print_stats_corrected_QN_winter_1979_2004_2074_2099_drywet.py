"""Flood statistics for Quinta Normal corrected data.

This script computes the number of days with Q > 500m3/s for observed and 
modeled data, as well as the number of days with rprecipitation > 3mm. The
modeled data is corrected using quantile mapping.
"""

import numpy as np
import xarray as xr
from scipy.stats import gamma

import discharge_isolines
import quantile_mapping

minval = 3

# read observed data
fp = '/home/tcarrasco/result/data/floods/obs_QN_ERA5_1976_2004.nc'
ds = xr.open_dataset(fp)
pr = ds['pr'].sel(time=slice('1979-01-01', '2004-12-31'))
H0 = ds['H0'].sel(time=slice('1979-01-01', '2004-12-31'))

obs_pr = pr.where(pr.time.dt.month.isin([5, 6, 7, 8, 9]), drop=True)
obs_H0 = H0.where(H0.time.dt.month.isin([5, 6, 7, 8, 9]), drop=True)

# read modeled data - historical period
fp = '/home/tcarrasco/result/data/floods/mod_QN_CMIP5_1976_2004.nc'
ds = xr.open_dataset(fp)
ds = ds.sel(time=slice('1979-01-01', '2004-12-31'))
ds = ds.where(ds.time.dt.month.isin([5, 6, 7, 8, 9]), drop=True)

models = ds['model'].values
mod_data = {}

# unpack and filter data with pr > 3mm
for model in models:
    mod_H0 = ds['H0'].sel(model=model)
    mod_pr = ds['pr'].sel(model=model)
    mod_data[model] = {'H0_hist': mod_H0, 'pr_hist': mod_pr}
    
# read modeled data - future period
fp = '/home/tcarrasco/result/data/floods/mod_QN_CMIP5_2071_2099.nc'
ds = xr.open_dataset(fp)
ds = ds.sel(time=slice('2074-01-01', '2099-12-31'))
ds = ds.where(ds.time.dt.month.isin([5, 6, 7, 8, 9]), drop=True)

# unpack and filter data with pr > 3mm
for model in models:
    mod_H0 = ds['H0'].sel(model=model)
    mod_pr = ds['pr'].sel(model=model)
    mod_data[model].update({'H0_future': mod_H0, 'pr_future': mod_pr})

# discharge isolines
fun_isoline_500m3s = discharge_isolines.gen_isoline_500m3s()

def compute_N500(pr, H0):
    """Compute the number of days with Q > 500m3/s.
    
    Parameters
    ----------
    pr : xarray.DataArray
        Precipitation data.
    H0 : xarray.DataArray
        Isotherm 0C data.

    Returns
    -------
    N500 : float
        Number of days with Q > 500m3/s.
    """
    
    N500 = 0
    for i in range(pr.size):
        try:
            if H0[i] > fun_isoline_500m3s(pr[i]):
                N500 += 1
        except:
            pass
    return N500

for model in models:
    
    pr_hist = mod_data[model]['pr_hist']
    H0_hist = mod_data[model]['H0_hist']
    
    pr_future = mod_data[model]['pr_future']
    H0_future = mod_data[model]['H0_future']
    
    # correct modeled data using quantile mapping and plot
    transfer_fun_pr = quantile_mapping.gen_transfer_fun_precip_drywet(obs_pr, pr_hist, minval)
    transfer_fun_H0 = quantile_mapping.gen_transfer_fun_H0(obs_H0, H0_hist)
    
    pr_hist_corrected = transfer_fun_pr(pr_hist)
    H0_hist_corrected = transfer_fun_H0(H0_hist)
    
    pr_future_corrected = transfer_fun_pr(pr_future)
    H0_future_corrected = transfer_fun_H0(H0_future)
    
    plot_pr_hist_corrected = pr_hist_corrected[pr_hist_corrected > minval]
    plot_H0_hist_corrected = H0_hist_corrected[pr_hist_corrected > minval]
    
    plot_pr_future_corrected = pr_future_corrected[pr_future_corrected > minval]
    plot_H0_future_corrected = H0_future_corrected[pr_future_corrected > minval]

    
    print(model, 'Ntot hist', pr_hist_corrected.size, H0_hist_corrected.size)
    print(model, 'Nwet hist', plot_pr_hist_corrected.size)
    print(model, 'N500 hist', compute_N500(pr_hist_corrected, H0_hist_corrected))
    
    print(model, 'Ntot future', pr_future_corrected.size, H0_future_corrected.size)
    print(model, 'Nwet future', plot_pr_future_corrected.size)
    print(model, 'N500 future', compute_N500(pr_future_corrected, H0_future_corrected))
    
    print(model, 'MEAN(PRECIP > 3) hist - Future', int(plot_pr_hist_corrected.mean()), 
          int(plot_pr_future_corrected.mean()))
    print(model, 'MEAN(H0[Precip > 3]) hist - Future', int(plot_H0_hist_corrected.mean()), 
          int(plot_H0_future_corrected.mean()))
    
    cdf = quantile_mapping.cdf_fun_precip_drywet
    invcdf = quantile_mapping.invcdf_fun_precip_drywet
    
    ev_hist = invcdf(0.99, pr_hist_corrected, minval)
    ev_future = invcdf(0.99, pr_future_corrected, minval)
    
    print(model, 'EV hist - Future', int(ev_hist), int(ev_future))
    
    tau_ev_hist = 1/(1-cdf(35, pr_hist_corrected, minval))
    tau_ev_future = 1/(1-cdf(35, pr_future_corrected, minval))
    
    print(model, 'TAU 35mm hist - Future', int(tau_ev_hist), 
          int(tau_ev_future))
    
    