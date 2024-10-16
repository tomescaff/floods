"""Precipitation vs. H0 scatter plot for Quinta Normal station

This script reads the observed precipitation and H0 data from the Quinta Normal
station and creates a scatter plot of the precipitation vs. H0. The observed
data is from 1976 to 2004. The H0 data is computed from ERA5 data at the
closest grid point to the Quinta Normal station. The H0 data is computed using
the temperature at 500 hPa and the geopotential height at 500 hPa. The
temperature is converted to height using the a lapse rate of 6.5 ºC/km. The
H0 is computed as the height at which the temperature is 0 ºC. Precipitation
data is from the Quinta Normal station.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

import discharge_isolines

# read data
fp = '/home/tcarrasco/result/data/floods/obs_QN_ERA5_1976_2004.nc'
ds = xr.open_dataset(fp)

pr = ds['pr'].sel(time=slice('1976-01-01', '2004-12-31'))
H0 = ds['H0'].sel(time=slice('1976-01-01', '2004-12-31'))

pr = pr.where(pr.time.dt.month.isin([5, 6, 7, 8, 9]), drop=True)
H0 = H0.where(H0.time.dt.month.isin([5, 6, 7, 8, 9]), drop=True)

# filter data with pr > 3mm
H0 = H0.where(pr > 3, drop=True)
pr = pr.where(pr > 3, drop=True)

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

fig = plt.figure(figsize=(9,9))
x = np.linspace(10, 300, 100)
iso500 = fun_isoline_500m3s(x)
plt.scatter(pr, H0, c='grey', s=pr, alpha=0.65, label='Observed (1979-2004)')
plt.plot(x, iso500, 'b', lw=3, label='Q=500 m3/s')
plt.ylim(500, 5000)
plt.xlim(0, 120)
plt.xlabel('Precipitation (mm/day)')
plt.ylabel('H0 (m)')
plt.grid(ls='--', lw=0.4, c='grey')
plt.legend()
plt.tick_params(axis="both", direction="in")
plt.tight_layout()

print('Printing min-max...')
print(f'Precip: [{pr.min().values:.0f}, {pr.max().values:.0f}]')
print(f'H0: [{H0.min().values:.0f}, {H0.max().values:.0f}]')

bd = '/home/tcarrasco/result/images/floods/'
fn = f'pr_h0_obs_QN.png'
plt.savefig(bd + fn, format='png', dpi=300)


