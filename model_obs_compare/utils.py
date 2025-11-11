import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
import xarray as xr
import os
import datetime
from scipy.interpolate import interp1d

def standard_value(ds, var, fillv=None):
    # if fillv is not given, get '_FillValue' from ds[var.attrs]
    if fillv is None:
        fillv = ds[var].attrs['_FillValue'][0]
    data = ds[var][:].astype(float)
    # data = ds[var][:]
    data[np.where(data==fillv)] = np.nan
    return data

def get_pressure_tropomi(psurface,etaa,etab):
    original = psurface
    # expand matrix using np.repeat
    # Setting the axis parameter to -1 means repeating along the last axis (i.e., the third dimension)
    expanded = np.repeat(original[:, :, np.newaxis], 34, axis=-1)
    # calculate pressure at each model layer edge
    pedge = etaa + etab * expanded
    return pedge

def convert_bytes_to_datetimes(data):
    """
    Convert a NumPy array containing ISO 8601 formatted datetime byte strings to an array of datetime objects.
    Parameters:
        data (numpy.ndarray): numpy array containing datetime byte strings
    Returns:
        numpy.ndarray: numpy array containing datetime objects.
    """
    def bytes_to_datetime(byte_string):
        """convert single byte string to datetime object"""
        string = byte_string.decode('utf-8')  # decode to regular string
        return datetime.datetime.strptime(string, '%Y-%m-%dT%H:%M:%S.%fZ')

    # apply transformation function
    vectorized_converter = np.vectorize(bytes_to_datetime)

    return vectorized_converter(data)

def get_valid_tropomi(data,error,flag,mask1):
    for key in data.keys():
        data[key] = data[key][mask1]
    for key in error.keys():
        error[key] = error[key][mask1]
    for key in flag.keys():
        flag[key] = flag[key][mask1]
    return data,error,flag

def regrid_sw_vertically(y,x,xnew):
    f = interp1d(x,y,kind='linear',fill_value="extrapolate")
    ynew = f(xnew)
    # there might be pxSW_GC<0, probably when 
    # surface_pressure of GC > that of TROPOMI
    ynew[ynew < 0] = 0.001
    return ynew

def cal_avg(data):
    # convert 0 to nan
    for key in data.keys():
        data[key][data[key] == 0] = np.nan
    
    # convert 0 to nan
    for key,v in data.items():
        if ('num' in key):
            continue
        else:
            data[key] = data[key] / data['num']
    return data

def save_output_grid_variables_monthly(gridtimes,gridlats,gridlons,data,path):
    # get monthly average for each grid cell
    data = cal_avg(data)
    
    species_vcd = xr.Dataset({k: (['time','lat', 'lon'], v) for k, v in data.items() if v.ndim == 3},
                         coords={'time':gridtimes, 'lat': gridlats, 'lon': gridlons})
    
    if os.path.exists(path):
        os.remove(path)
    species_vcd.to_netcdf(path)

    return species_vcd