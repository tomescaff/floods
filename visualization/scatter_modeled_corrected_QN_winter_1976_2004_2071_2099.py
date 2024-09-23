"""Precip. vs. H0 scatter plot for Quinta Normal modeled and corrected data.

This script plots the modeled and corrected data of precipitation and H0 from 
several CMIP5 climate models at the closest grid point to Quinta Normal station 
for the historical period 1976-2004 and the future period 2071-2099 under the 
RCP8.5 scenario. The H0 modeled data is computed using the temperature at 500 
hPa, the geopotential height at 500 hPa, and a lapse rate of 6.5 ºC/km. The H0 
is computed as the height at which the temperature is 0 ºC. Both variables are
corrected using quantile mapping on the observed data from the Quinta Normal.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

import discharge_isolines
import quantile_mapping

# read observed data
fp = '/home/tcarrasco/result/data/floods/obs_QN_ERA5_1976_2004.nc'
ds = xr.open_dataset(fp)

pr = ds['pr'].sel(time=slice('1976-01-01', '2004-12-31'))
H0 = ds['H0'].sel(time=slice('1976-01-01', '2004-12-31'))
pr = pr.where(pr.time.dt.month.isin([5, 6, 7, 8, 9]), drop=True)
H0 = H0.where(H0.time.dt.month.isin([5, 6, 7, 8, 9]), drop=True)

# filter data with pr > 3mm
obs_H0 = H0.where(pr > 3, drop=True)
obs_pr = pr.where(pr > 3, drop=True)

# read modeled data - historical period
fp = '/home/tcarrasco/result/data/floods/mod_QN_CMIP5_1976_2004.nc'
ds = xr.open_dataset(fp)
ds = ds.sel(time=slice('1976-01-01', '2004-12-31'))
ds = ds.where(ds.time.dt.month.isin([5, 6, 7, 8, 9]), drop=True)

models = ds['model'].values
mod_data = {}

# unpack and filter data with pr > 3mm
for model in models:
    mod_H0 = ds['H0'].sel(model=model)
    mod_pr = ds['pr'].sel(model=model)
    mod_H0 = mod_H0.where(mod_pr > 3, drop=True)
    mod_pr = mod_pr.where(mod_pr > 3, drop=True)
    mod_data[model] = {'H0_hist': mod_H0, 'pr_hist': mod_pr}
    
# read modeled data - future period
fp = '/home/tcarrasco/result/data/floods/mod_QN_CMIP5_2071_2099.nc'
ds = xr.open_dataset(fp)
ds = ds.sel(time=slice('2071-01-01', '2099-12-31'))
ds = ds.where(ds.time.dt.month.isin([5, 6, 7, 8, 9]), drop=True)

# unpack and filter data with pr > 3mm
for model in models:
    mod_H0 = ds['H0'].sel(model=model)
    mod_pr = ds['pr'].sel(model=model)
    mod_H0 = mod_H0.where(mod_pr > 3, drop=True)
    mod_pr = mod_pr.where(mod_pr > 3, drop=True)
    mod_data[model].update({'H0_future': mod_H0, 'pr_future': mod_pr})

# discharge isolines
fun_isoline_500m3s = discharge_isolines.gen_isoline_500m3s()
x = np.linspace(10, 300, 100)
iso500 = fun_isoline_500m3s(x)

# basic plot settings
font_dir = ['/home/tcarrasco/result/fonts/Merriweather',
            '/home/tcarrasco/result/fonts/arial']
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)
plt.rcParams['font.family'] = 'arial'
plt.rcParams['font.size'] = 17
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

fig, axs = plt.subplots(len(models),2, figsize=(15,30))

for model in models:
    i = models.tolist().index(model)*2
    ax1 = axs.flatten()[i]
    ax2 = axs.flatten()[i+1]
    
    pr_hist = mod_data[model]['pr_hist']
    H0_hist = mod_data[model]['H0_hist']
    
    pr_future = mod_data[model]['pr_future']
    H0_future = mod_data[model]['H0_future']
    
    # plot modeled data as is
    plt.sca(ax1)
    plt.title(model)
    plt.scatter(pr_hist, H0_hist, c='grey', s=pr_hist, alpha=0.65, 
                label='Hist. (1979-2004)')
    plt.scatter(pr_future, H0_future, c='red', s=pr_future, alpha=0.65, 
                label='RCP8.5 (2071-2099)')
    plt.plot(x, iso500, 'b', lw=3, label='Q=500 m3/s')
    plt.ylim(500, 5000)
    plt.xlim(0, 350)
    plt.xlabel('Precipitation (mm/day)')
    plt.ylabel('H0 (m)')
    plt.grid(ls='--', lw=0.4, c='grey')
    plt.legend()
    plt.tick_params(axis="both", direction="in")
    
    # correct modeled data using quantile mapping and plot
    transfer_fun_pr = quantile_mapping.gen_transfer_fun_precip(obs_pr, pr_hist)
    transfer_fun_H0 = quantile_mapping.gen_transfer_fun_H0(obs_H0, H0_hist)
    
    plt.sca(ax2)
    plt.title(model + ' (corrected)')
    plt.scatter(transfer_fun_pr(pr_hist), transfer_fun_H0(H0_hist), 
                s=transfer_fun_pr(pr_hist), alpha=0.65, c='grey', 
                label='Hist. (1979-2004)')
    plt.scatter(transfer_fun_pr(pr_future), transfer_fun_H0(H0_future),  
                s=transfer_fun_pr(pr_future), alpha=0.65, c='red',
                label='RCP8.5 (2071-2099)')
    plt.plot(x, iso500, 'b', lw=3, label='Q=500 m3/s')
    plt.ylim(500, 5000)
    plt.xlim(0, 120)
    plt.xlabel('Precipitation (mm/day)')
    plt.ylabel('H0 (m)')
    plt.grid(ls='--', lw=0.4, c='grey')
    plt.legend()
    plt.tick_params(axis="both", direction="in")

plt.tight_layout()

bd = '/home/tcarrasco/result/images/floods/'
fn = f'pr_h0_mod_QN.png'
plt.savefig(bd + fn, format='png', dpi=300)