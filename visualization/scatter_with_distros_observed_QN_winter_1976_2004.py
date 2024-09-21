"""Precipitation vs. H0 scatter plot for Quinta Normal with distributions

This script reads the observed precipitation and H0 data from the Quinta Normal
station and creates a scatter plot of the precipitation vs. H0. The observed
data is from 1976 to 2004. The H0 data is computed from ERA5 data at the
closest grid point to the Quinta Normal station. The H0 data is computed using
the temperature at 500 hPa and the geopotential height at 500 hPa. The
temperature is converted to height using the a lapse rate of 6.5 ºC/km. The
H0 is computed as the height at which the temperature is 0 ºC. Precipitation
data is from the Quinta Normal station.

This script also plots the distributions of precipitation and H0.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.gridspec import GridSpec
from scipy.stats import gamma, norm

import discharge_isolines

# read data
fp = '/home/tcarrasco/result/data/floods/obs_QN_ERA5_1976_2004.nc'
ds = xr.open_dataset(fp)

pr = ds['pr']
H0 = ds['H0']

# filter data with pr > 3mm
H0 = H0.where(pr > 3, drop=True)
pr = pr.where(pr > 3, drop=True)

normfit = norm.fit(H0.values)
gammafit = gamma.fit(pr.values, floc=3)

# discharge isolines
fun_isoline_500m3s = discharge_isolines.gen_isoline_500m3s()

# basic plot settings
font_dir = ['/home/tcarrasco/result/fonts/Merriweather',
            '/home/tcarrasco/result/fonts/arial']
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)
plt.rcParams['font.family'] = 'arial'
plt.rcParams['font.size'] = 17
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

fig = plt.figure(figsize=(11, 11))
gs = GridSpec(5, 5, figure=fig)

ax_main = fig.add_subplot(gs[1:, :-1])
plt.sca(ax_main)
x = np.linspace(10, 300, 100)
iso500 = fun_isoline_500m3s(x)
plt.scatter(pr, H0, c='grey', s=pr*1.5, alpha=0.65, label='Observed (1979-2004)')
plt.plot(x, iso500, 'b', lw=3, label='Q=500 m3/s')
plt.ylim(500, 5000)
plt.xlim(0, 120)
plt.xlabel('Precipitation (mm/day)')
plt.ylabel('H0 (m)')
plt.grid(ls='--', lw=0.4, c='grey')
plt.legend()
plt.tick_params(axis="both", direction="in")

ax_prdist = fig.add_subplot(gs[0, :-1])
plt.sca(ax_prdist)
x = np.linspace(3, 120, 100)
y = gamma.pdf(x, *gammafit)
plt.plot(x, y, 'r', lw=2)
bins = np.arange(3,120,3)
plt.hist(pr, bins=bins, density=True, fc='grey', ec='k', # type: ignore
         alpha=0.65) 
plt.xlim(0, 120)
plt.ylim(0, 0.08)
plt.grid(ls='--', lw=0.4, c='grey')
plt.tick_params(axis="both", direction="in")
plt.gca().set_xticklabels([])
plt.ylabel('PDF')
plt.yticks([0, 4e-2, 8e-2], ['0', '4e-2', '8e-2'])
plt.legend(['Gamma fit', 'Data'])

ax_h0dist = fig.add_subplot(gs[1:, -1])
plt.sca(ax_h0dist)
x = np.linspace(500, 5000, 100)
y = norm.pdf(x, *normfit)
plt.plot(y, x, 'r', lw=2)
bins = np.arange(500,5000,100)
plt.hist(H0, bins=bins, density=True, fc='grey', ec='k', # type: ignore
         alpha=0.65, orientation="horizontal") 
plt.xlim(0, 8e-4)
plt.ylim(500, 5000)
plt.grid(ls='--', lw=0.4, c='grey')
plt.tick_params(axis="both", direction="in")
plt.gca().set_yticklabels([])
plt.xlabel('PDF')
plt.xticks([0, 4e-4, 8e-4], ['0', '4e-4', '8e-4'])
plt.legend(['Normal fit', 'Data'])

plt.tight_layout()

print('Printing min-max...')
print(f'Precip: [{pr.min().values:.0f}, {pr.max().values:.0f}]')
print(f'H0: [{H0.min().values:.0f}, {H0.max().values:.0f}]')

bd = '/home/tcarrasco/result/images/floods/'
fn = f'pr_h0_obs_QN_distros.png'
plt.savefig(bd + fn, format='png', dpi=300)


