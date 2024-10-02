"""Statistics of observed daily precipitation at Quinta Normal station.

This script reads the observed daily precipitation data at the Quinta Normal
station and creates a histogram of the data. The script also fits a gamma
distribution to the data and plots the PDF and CDF of the data. The script also
reads the modeled daily precipitation data from CMIP5 and perform a quantile
mapping to correct the modeled data. The script creates a transfer function to
correct the modeled data and plots the transfer function. 

This script considers dry days in the fitting procedure.
"""

import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.signal import resample

def cdf(x, dist_full, minval):
    """Computes the CDF of a value x in a distribution dist_full.
    
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
    if x < minval:
        cdf = np.nan
    elif x == minval:
        cdf = cdf_minval
    else:
        cdf_wet = gamma.cdf(x, *gamma.fit(dist_wet, floc=minval))
        cdf = cdf_minval + (1-cdf_minval)*cdf_wet
    return cdf

cdf_fun = np.vectorize(cdf, excluded=[1, 2])

# set parameters
minval = 3 # lower threshold in mm
x = np.arange(0, 350, 3.01)

# read observed data at Quinta Normal station
fp = '/home/tcarrasco/result/data/floods/obs_QN_ERA5_1976_2004.nc'
ds = xr.open_dataset(fp)
pr = ds['pr'].sel(time=slice('1979-01-01', '2004-12-31'))
obs_full = pr.where(pr.time.dt.month.isin([5, 6, 7, 8, 9]), drop=True)
obs_filt = obs_full.where(obs_full > minval, drop=True)

# read modeled data from CMIP5
fp = '/home/tcarrasco/result/data/floods/mod_QN_CMIP5_1976_2004.nc'
ds = xr.open_dataset(fp)
ds = ds.sel(time=slice('1976-01-01', '2004-12-31'))
ds = ds.where(ds.time.dt.month.isin([5, 6, 7, 8, 9]), drop=True)
pr = ds['pr'].isel(model=0)
mod_full = pr.where(pr.time.dt.month.isin([5, 6, 7, 8, 9]), drop=True)
mod_filt = mod_full.where(mod_full > minval, drop=True)

# cdf obs and mod
cdf_obs = cdf_fun(x, obs_full, minval)
cdf_mod = cdf_fun(x, mod_full, minval)

def invcdf_obs(p, dist_full, minval):
    """Inverse CDF of the observed data.
    
    Params
    ------
    p: float
        Probability value
    """
    cdf_minval = cdf(minval, dist_full, minval)
    if p < cdf_minval:
        return np.nan
    elif p == cdf_minval:
        return minval
    else:
        dist_wet = dist_full[dist_full > minval]
        gamfit_obs = gamma.fit(dist_wet, floc=minval)
        z = gamma.ppf((p - cdf_minval)/(1-cdf_minval), *gamfit_obs)
        return z

p90_obs = invcdf_obs(0.9, obs_full, minval)
p99_obs = invcdf_obs(0.99, obs_full, minval)
tau_60mm = 1/(1-cdf(60, obs_full, minval))   
# plot data
fig, axs = plt.subplots(4, 2, figsize=(10, 20))

# plot histogram with observed data
plt.sca(axs[0,0])
hist, bins = np.histogram(obs_full, x, density=True)
center = (bins[:-1] + bins[1:]) / 2
cdf_bar = np.cumsum(hist)*np.diff(bins)
plt.plot(center, cdf_bar, c='k', lw=2)
plt.plot(x, cdf_obs , 'r--', lw=2)
plt.axvline(p90_obs, c='fuchsia', lw=1) # type: ignore
plt.axvline(p99_obs, c='fuchsia', ls='--', lw=1) # type: ignore
plt.title('QN daily precipitation')
plt.legend(['Mixed Gamma fit', 'Data', f'P90 ({p90_obs:.0f} mm/day)', 
            f'P99 ({p99_obs:.0f} mm/day)'])
plt.xlabel('Precipitation (mm/day)')
plt.ylabel('CDF')
plt.xlim([0, 350])
plt.ylim([0.65, 1.01])

# plot histogram with modeled data
plt.sca(axs[0,1])
hist, bins = np.histogram(mod_full, x, density=True)
center = (bins[:-1] + bins[1:]) / 2
cdf_bar = np.cumsum(hist)*np.diff(bins)
plt.plot(center, cdf_bar, c='k', lw=2)
plt.plot(x, cdf_mod, 'b--', lw=2)
plt.title('Model daily precipitation')
plt.legend(['Mixed Gamma fit', 'Data'])
plt.xlabel('Precipitation (mm/day)')
plt.ylabel('CDF')
plt.xlim([0, 350])
plt.ylim([0.65, 1.01])

# plot CDF - observed and modeled
plt.sca(axs[1,0])
plt.plot(x, cdf_obs, 'r-', lw=2)
plt.plot(x, cdf_mod, 'b-', lw=2)
plt.title('CDF')
plt.legend(['Obs', 'Mod'])
plt.xlabel('Precipitation (mm/day)')
plt.ylabel('CDF')
plt.xlim([0, 350])
plt.ylim([0.65, 1.01])

# define transfer function: mod -> obs
def transfer(z):
    p = cdf_fun(z, mod_full, minval)
    h0 = cdf_fun(minval, obs_full, minval)
    if p <= h0:
        return minval
    else:
        return gamma.ppf( (p-h0)/(1-h0), *gamma.fit(obs_filt, floc=minval))
transfer_fun = np.vectorize(transfer)

# plot transfer function
plt.sca(axs[1,1])
plt.plot(x, transfer_fun(x), 'green', lw=2)
plt.plot([0, 350], [0, 350], 'k--', lw=1)
plt.title('Transfer function y=f(x)')
plt.xlabel('Model precipitation (mm/day)')
plt.ylabel('Corrected precipitation (mm/day)')
plt.xlim([0, 350])
plt.ylim([0, 350])
plt.legend(['Transfer function', '1:1 line'])

# plot histogram of corrected data and gamma fit
plt.sca(axs[2,0])
mod_full_corrected = transfer_fun(mod_full)
hist_c, bins_c = np.histogram(mod_full_corrected, x, density=True)
center_c = (bins_c[:-1] + bins_c[1:]) / 2
cdf_bar_c = np.cumsum(hist_c)*np.diff(bins_c)
plt.plot(center_c, cdf_bar_c, c='k', lw=2)
plt.plot(x, cdf_obs, 'red', ls='--', lw=2)
plt.title('Model corrected daily precipitation')
plt.legend(['Gamma fit - model corrected', 'Gamma fit - observed'])
plt.xlabel('Precipitation (mm/day)')
plt.ylabel('CDF')
plt.xlim([0, 150])
plt.ylim([0.65, 1.01])

# plot annual time series
plt.sca(axs[2,1])
plt.title('Time series')
mod_full_annual = mod_full.resample(time='1YS').sum('time')
time = mod_full_annual.time.dt.year
da_mod_full_corrected = xr.DataArray(transfer_fun(mod_full), 
                                     coords={'time': mod_full.time})
mod_full_corrected_annual = da_mod_full_corrected.resample(time='1YS')
mod_full_corrected_annual = mod_full_corrected_annual.sum('time')
obs_full_annual = obs_filt.resample(time='1YS').sum('time')
plt.plot(time, mod_full_annual, c='k')
plt.plot(time, mod_full_corrected_annual, c='green')
plt.plot(obs_full_annual.time.dt.year, obs_full_annual, c='red')
plt.xlabel('Time')
plt.ylabel('Annual acc. precipitation (mm/day) ')
plt.legend(['Model', 'Bias corrected', 'OBS'])

# # frequency plot
# plt.sca(axs[3,0])
# plt.title('Frequency plot')
# plt.hist([mod_filt, transfer(mod_filt), obs_filt], 
#          bins=np.arange(minval, 270, 20), color=['b', 'green', 'r'], alpha=0.6, # type: ignore
#          ec='k', lw=0.3, density=False) 
# plt.xlabel('Precipitation (mm/day) [every 20 mm]')
# plt.ylabel('Frequency')
# nobs_dry = np.count_nonzero(obs_full.values <= minval)
# nmod_dry = np.count_nonzero(mod_full.values <= minval)
# nmod_corr_dry = np.count_nonzero(transfer(mod_full) <= minval)
# plt.legend([f'Model ({nmod_dry} dry days)', 
#             f'Bias corrected ({nmod_corr_dry} dry days)', 
#             f'OBS ({nobs_dry} dry days)'])
# plt.ylim([0, 1000])

# # experiment 
# plt.sca(axs[3,1])
# N = 10000
# plt.title(f'Experiment: resample N={N}')
# obs_filt_resampled = resample(obs_filt, N)
# mod_filt_resampled = resample(mod_filt, N)
# mod_filt_corrected_resampled = resample(transfer(mod_filt), N)
# plt.hist([mod_filt_resampled, 
#           mod_filt_corrected_resampled, 
#           obs_filt_resampled], 
#          bins=np.arange(minval, 270, 20), color=['b', 'green', 'r'], alpha=0.6, # type: ignore
#          edgecolor='k', lw=0.3, density=True) 
# plt.xlabel('Precipitation (mm/day) [every 20 mm]')
# plt.ylabel('Frequency')
# plt.legend(['Model', 'Bias corrected', 'OBS'])

plt.tight_layout()

bd = '/home/tcarrasco/result/images/floods/'
fn = 'stats_QN_daily_precip_drywet.png'
plt.savefig(bd + fn, format='png', dpi=500)