"""Generate isolines of discharge values in the precipitation-H0 plane."""

from scipy import interpolate

# isolines of 500 m3/s discharge valid for Central Chile 
# [daily precipitation (mm), H0 (m)]
xy_isoline_500m3s = [[10, 5000],
                     [12, 4000], 
                     [15, 3500], 
                     [20, 3000], 
                     [50, 2450],
                     [100, 2000], 
                     [150, 1800],
                     [300, 1600]]

def gen_isoline_500m3s():
    x_points = [z[0] for z in xy_isoline_500m3s]
    y_points = [z[1] for z in xy_isoline_500m3s]
    fun_interp = interpolate.interp1d(x_points, y_points)
    return fun_interp