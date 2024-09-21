"""Statistics of observed daily precipitation at Quinta Normal station.

This script reads the observed daily precipitation data at the Quinta Normal
station and creates a histogram of the data. The script also fits a gamma
distribution to the data and plots the PDF and CDF of the data. The script also
reads the modeled daily precipitation data from CMIP5 and perform a quantile
mapping to correct the modeled data. The script creates a transfer function to
correct the modeled data and plots the transfer function. 
"""

import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.signal import resample

# set parameters
minval = 3 # lower threshold in mm
xbins = np.arange(0, 350, 0.5)
x = xbins[xbins>minval]

# read observed data at Quinta Normal station
df = pd.read_csv('QN_daily_precip.csv', 
                 parse_dates={'time': ['agno', ' mes', ' dia']}, )
df = df.set_index('time')
da = df[' valor'].to_xarray()
da = da.sel(time=slice('1976-01-01', '2004-12-31'))
obs_full = da.where(da.time.dt.month.isin([5, 6, 7, 8, 9]), drop=True)
obs_filt = obs_full.where(obs_full > minval, drop=True)

# read modeled data from CMIP6
ds = xr.open_dataset('iso0C_data_H0_v1.nc')
da = ds['pr']
da = da.sel(time=slice('1976-01-01', '2004-12-31'))
da = da.where(da.time.dt.month.isin([5, 6, 7, 8, 9]), drop=True)
mod_full = da[0,:] # model 0
mod_filt = mod_full.where(mod_full > minval, drop=True)

# plot data
fig, axs = plt.subplots(4, 2, figsize=(10, 20))

# plot histogram with observed data
plt.sca(axs[0,0])
hist, bins = np.histogram(obs_filt, xbins, density=True)
fullwidth = (bins[1] - bins[0])
width = 1.0 * fullwidth
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width, ec='k', fc='b', alpha=0.6, 
        lw=0.3)
gamfit_obs_filt = gamma.fit(obs_filt, floc=minval)
plt.plot(x, gamma.pdf(x, *gamfit_obs_filt), 'r-', lw=2)
plt.title('QN daily precipitation')
plt.legend(['Gamma fit', 'Data'])
plt.xlabel('Precipitation (mm/day)')
plt.ylabel('Density (PDF)')
plt.xlim([0, 350])
plt.ylim([0, 0.12])

# plot histogram with modeled data
plt.sca(axs[0,1])
hist, bins = np.histogram(mod_filt, xbins, density=True)
fullwidth = (bins[1] - bins[0])
width = 1.0 * fullwidth
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width, ec='k', fc='b', alpha=0.6, 
        lw=0.3)
gamfit_mod_filt = gamma.fit(mod_filt, floc=minval)
plt.plot(x, gamma.pdf(x, *gamfit_mod_filt), 'k-', lw=2)
plt.title('Model daily precipitation')
plt.legend(['Gamma fit', 'Data'])
plt.xlabel('Precipitation (mm/day)')
plt.ylabel('Density (PDF)')
plt.xlim([0, 350])
plt.ylim([0, 0.12])

# plot CDF - observed and modeled
plt.sca(axs[1,0])
cdf_obs = gamma.cdf(x, *gamfit_obs_filt)
cdf_mod = gamma.cdf(x, *gamfit_mod_filt)
plt.plot(x, cdf_obs, 'r-', lw=2)
plt.plot(x, cdf_mod, 'k-', lw=2)
plt.title('CDF')
plt.legend(['Obs', 'Mod'])
plt.xlabel('Precipitation (mm/day)')
plt.ylabel('CDF')
plt.xlim([0, 350])
plt.ylim([0, 1])

# define transfer function
def transfer_fun(z):
    if z <= minval:
        return z
    else:
        return gamma.ppf(gamma.cdf(z, *gamfit_mod_filt), *gamfit_obs_filt)
transfer = np.vectorize(transfer_fun)

# plot transfer function
plt.sca(axs[1,1])
plt.plot(xbins, transfer(xbins), 'green', lw=2)
plt.plot([0, 350], [0, 350], 'k--', lw=1)
plt.title('Transfer function y=f(x)')
plt.xlabel('Model precipitation (mm/day)')
plt.ylabel('Corrected precipitation (mm/day)')
plt.xlim([0, 350])
plt.ylim([0, 350])
plt.legend(['Transfer function', '1:1 line'])

# plot histogram of corrected data and gamma fit
plt.sca(axs[2,0])
mod_filt_corrected = transfer(mod_filt)
hist, bins = np.histogram(mod_filt_corrected, xbins, density=True)
fullwidth = (bins[1] - bins[0])
width = 1.0 * fullwidth
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width, ec='k', fc='b', alpha=0.6, 
        lw=0.3)
gamfit_mod_filt_corrected = gamma.fit(mod_filt_corrected, floc=minval)
plt.plot(x, gamma.pdf(x, *gamfit_mod_filt_corrected), 'green', lw=2)
plt.plot(x, gamma.pdf(x, *gamfit_obs_filt), 'red', ls='--', lw=2)
plt.title('Model corrected daily precipitation')
plt.legend(['Gamma fit - model corrected', 'Gamma fit - observed', 'Data'])
plt.xlabel('Precipitation (mm/day)')
plt.ylabel('Density (PDF)')
plt.xlim([0, 150])
plt.ylim([0, 0.12])

# plot annual time series
plt.sca(axs[2,1])
plt.title('Time series')
mod_filt_annual = mod_filt.resample(time='1YS').mean('time')
time = mod_filt_annual.time.dt.year
da_mod_filt_corrected = xr.DataArray(mod_filt_corrected, 
                                     coords={'time': mod_filt.time})
mod_filt_corrected_annual = da_mod_filt_corrected.resample(time='1YS')
mod_filt_corrected_annual = mod_filt_corrected_annual.mean('time')
obs_filt_annual = obs_filt.resample(time='1YS').mean('time')
plt.plot(time, mod_filt_annual, c='k')
plt.plot(time, mod_filt_corrected_annual, c='green')
plt.plot(time, obs_filt_annual, c='red')
plt.xlabel('Time')
plt.ylabel('Annual mean precipitation [wet days only (>3mm)] (mm/day) ')
plt.legend(['Model', 'Bias corrected', 'OBS'])

# frequency plot
plt.sca(axs[3,0])
plt.title('Frequency plot')
plt.hist([mod_filt, transfer(mod_filt), obs_filt], 
         bins=np.arange(minval, 270, 20), color=['b', 'green', 'r'], alpha=0.6, # type: ignore
         ec='k', lw=0.3, density=False) 
plt.xlabel('Precipitation (mm/day) [every 20 mm]')
plt.ylabel('Frequency')
nobs_dry = np.count_nonzero(obs_full.values <= minval)
nmod_dry = np.count_nonzero(mod_full.values <= minval)
nmod_corr_dry = np.count_nonzero(transfer(mod_full) <= minval)
plt.legend([f'Model ({nmod_dry} dry days)', 
            f'Bias corrected ({nmod_corr_dry} dry days)', 
            f'OBS ({nobs_dry} dry days)'])
plt.ylim([0, 1000])

# experiment 
plt.sca(axs[3,1])
N = 10000
plt.title(f'Experiment: resample N={N}')
obs_filt_resampled = resample(obs_filt, N)
mod_filt_resampled = resample(mod_filt, N)
mod_filt_corrected_resampled = resample(transfer(mod_filt), N)
plt.hist([mod_filt_resampled, 
          mod_filt_corrected_resampled, 
          obs_filt_resampled], 
         bins=np.arange(minval, 270, 20), color=['b', 'green', 'r'], alpha=0.6, # type: ignore
         edgecolor='k', lw=0.3, density=True) 
plt.xlabel('Precipitation (mm/day) [every 20 mm]')
plt.ylabel('Frequency')
plt.legend(['Model', 'Bias corrected', 'OBS'])

plt.tight_layout()
plt.savefig('stats_QN_daily_precip.png', dpi=300)