import glob
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann as kB

def load_planeflight_ATom(rundir, campaign, region=None):
    '''
    Reads in GEOS-Chem sampled output planeflight.log files (ATom BY CAMPAIGN). 
    For ATom, groups chemical species by latitude (0-30 N/S) and pressure bins.

    Args:
        rundir: directory containing ATom planeflight.log files
        campaign: 1, 2, 3, or 4
        region: 'Pacific' or 'Atlantic', for ATom

    Returns:
        grouped DataFrame with mean values (OHR: s-1, OH: molec/cm3, CO: ppb, temp: K) in pressure bins.
    '''

    files = glob.glob(f'{rundir}/ATom_logs/ATom{campaign}/plane.ATom.log.*')
    files.sort()

    # get header from first file
    with open(files[0], 'r') as f:
        header_line = f.readline().strip().split()

    cols = ['LAT', 'LON', 'PRESS', 'CO', 'OH', 'OHR_99999', 'GMAO_TEMP']
    usecols = [header_line.index(col) for col in cols]

    # Load all files
    all_dfs = []
    for f in files:
        df = pd.read_csv(
            f,
            delim_whitespace=True,
            header=0,
            usecols=usecols,
            names=header_line
        )
        all_dfs.append(df)

    # Combine all into one DataFrame
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # group by hemisphere and group pressure bins
    # df_NH = combined_df[combined_df['LAT'] > 0].copy()
    # df_SH = combined_df[combined_df['LAT'] < 0].copy()

    # group by ocean
    if region == 'Pacific': # longitude from 120 to -80
        combined_df = combined_df[(combined_df['LON'] > 120) | (combined_df['LON'] < -80)].copy()
    elif region == 'Atlantic': # longitude from -75 to 10
        combined_df = combined_df[combined_df['LON'].between(-75, 10)].copy()

    # group by 0-30N and 0-30S, 30-60N
    df_0to30N = combined_df[(combined_df['LAT'] > 0) & (combined_df['LAT'] < 30)].copy()
    df_0to30S = combined_df[(combined_df['LAT'] < 0) & (combined_df['LAT'] > -30)].copy()
    df_30to60N = combined_df[(combined_df['LAT'] > 30) & (combined_df['LAT'] < 60)].copy()

    df_0to30N['PRESS_bin'] = (df_0to30N['PRESS'] // 50) * 50
    df_0to30S['PRESS_bin'] = (df_0to30S['PRESS'] // 50) * 50
    df_30to60N['PRESS_bin'] = (df_30to60N['PRESS'] // 50) * 50

    # Get mean in each pressure bin
    df_0to30N_mean = (
        df_0to30N
        .groupby('PRESS_bin')
        .agg(
            OH_mean      = ('OH',        'mean'),
            OHR_mean     = ('OHR_99999', 'mean'),
            CO_mean      = ('CO',        'mean'),
            temp_mean    = ('GMAO_TEMP', 'mean'),
        )
        .reset_index()
    )

    df_0to30S_mean = (
        df_0to30S
        .groupby('PRESS_bin')
        .agg(
            OH_mean      = ('OH',        'mean'),
            OHR_mean     = ('OHR_99999', 'mean'),
            CO_mean      = ('CO',        'mean'),
            temp_mean    = ('GMAO_TEMP', 'mean'),
        )
        .reset_index()
    )

    df_30to60N_mean = (
        df_30to60N
        .groupby('PRESS_bin')
        .agg(
            OH_mean      = ('OH',        'mean'),
            OHR_mean     = ('OHR_99999', 'mean'),
            CO_mean      = ('CO',        'mean'),
            temp_mean    = ('GMAO_TEMP', 'mean'),
        )
        .reset_index()
    )

    return df_0to30N_mean, df_0to30S_mean, df_30to60N_mean


def load_planeflight_ATom_all(rundir):
    '''
    Reads in GEOS-Chem sampled output planeflight.log files (ALL ATom campaigns). 
    For ATom, groups chemical species by pressure bins.

    Args:
        rundir: directory containing ATom planeflight.log files

    Returns:
        grouped DataFrame with mean values (OHR: s-1, OH: molec/cm3, CO: ppb, temp: K) in pressure bins.
    '''

    files = glob.glob(f'{rundir}/ATom_logs/ATom*/*') # all logs from all campaigns
    files.sort()

    # get header from first file
    with open(files[0], 'r') as f:
        header_line = f.readline().strip().split()

    cols = ['YYYYMMDD', 'HHMM', 'LAT', 'LON', 'PRESS', 'CO', 'OH', 'OHR_99999', 'GMAO_TEMP']
    usecols = [header_line.index(col) for col in cols]

    # Load all files
    all_dfs = []
    for f in files:
        df = pd.read_csv(
            f,
            delim_whitespace=True,
            header=0,
            usecols=usecols,
            names=header_line
        )
        all_dfs.append(df)

    # Combine all into one DataFrame
    df_all_campaigns = pd.concat(all_dfs, ignore_index=True)

    # p_Pa = combined_df['PRESS'] * 100 # Pa
    # combined_df['n_air'] = (p_Pa / (kB * combined_df['GMAO_TEMP'])) * 1e-6 # molec/cm3

    return df_all_campaigns
    
    
def load_planeflight_KORUSAQ(rundir):
    '''
    Reads in GEOS-Chem sampled output planeflight.log files (KORUS-AQ). 
    For KORUS-AQ, groups by pressure bins only.

    Args:
        rundir: directory containing planeflight.log files

    Returns:
        grouped DataFrame with mean values (OHR: s-1, OH: molec/cm3, CO: ppb, temp: K) in pressure bins.
    '''

    files = glob.glob(f'{rundir}/plane.KORUSAQ.log.*')
    files.sort()

    # get header from first file
    with open(files[0], 'r') as f:
        header_line = f.readline().strip().split()

    cols = ['LAT', 'LON', 'PRESS', 'CO', 'OH', 'OHR_99999', 'GMAO_TEMP']
    usecols = [header_line.index(col) for col in cols]

    # Load all files
    all_dfs = []
    for f in files:
        df = pd.read_csv(
            f,
            delim_whitespace=True,
            header=0,
            usecols=usecols,
            names=header_line
        )
        all_dfs.append(df)

    # Combine all into one DataFrame
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # group by pressure bin only
    combined_df['PRESS_bin'] = (combined_df['PRESS'] // 50) * 50

    # Get mean in each pressure bin
    mean_df = (
        combined_df
        .groupby('PRESS_bin')
        .agg(
            OH_mean      = ('OH',        'mean'),
            OHR_mean     = ('OHR_99999', 'mean'),
            CO_mean      = ('CO',        'mean'),
            temp_mean    = ('GMAO_TEMP', 'mean'),
        )
        .reset_index()
    )

    return mean_df
        

### Pressure/altitude conversion functions are from Seb Eastham
### https://github.com/sdeastham/gcgridobj/blob/cec7427abc6701345056a7f862d1f6908672e425/gcgridobj/atmos_isa_mini.py
def altitude_to_many(z_m):
    '''
    Lightweight version of atmosisa from the Aerospace Toolbox
    atomsisa(height) implements International Standard Atmosphere values for temp, pressure, density (among others)
    
    Args:
        z_m: altitude (m)

    Returns:
        p_pa: pressure (Pa)
        T_K: temperature (K)
        rho_kgm3: air density (kg/m3)
    '''

    # Sort from lowest to highest altitude to march upward
    sort_idx = np.argsort(z_m)
    #z_sorted = z_m[sort_idx]
    
    # COESA data
    height_vec = np.array([-0.1,0,11,20,32,47,51,71,84.8520,1e6]) # km, bottom of each level
    lapse_vec = np.array([0.0,-6.5,0.0,1.0,2.8,0.0,-2.8,-2.0,0.0]) # K/km, lapse rate for each level
    T_reference = 288.15 # sea level
    height_delta = np.diff(height_vec) # each level's thickness

    # Change in temperature between each level
    T_delta = height_delta*lapse_vec
    T_vec = np.cumsum(np.concatenate([[T_reference],T_delta])) # cumulatively sum from sea level temp to get temperature at each level

    # Gas composition
    gas_MW = [28.0134,31.9988,39.948,44.00995,20.183,4.0026,83.80,131.30,16.04303,2.01594]

    gas_frac = [0.78084,0.209476,0.00934,0.000314,0.00001818,0.00000524,0.00000114,0.000000087,0.000002,0.0000005]

    # Normalize, to be 100% safe
    gas_frac = gas_frac/np.sum(gas_frac)

    R_star = 8.3144598 # universal gas constant J/K/mol
    g0 = 9.80665 # standard gravity, m/s2

    MgR = sum(gas_frac*gas_MW)*1e-3*g0/R_star # Mean molecular weight constant, converted to kg/mol
    
    n_vals = z_m.size
    p_pa = np.zeros(n_vals) # create pressure array

    # Temperature at target altitudes
    T_K = np.interp(z_m/1000.0,height_vec,T_vec) # linear interpolates the temperature profile onto each altitude

    # Now compute pressure at each level
    # Initial conditions
    iLo = 0
    iHi = 1
    zLo = height_vec[0] * 1000.0 # m
    zHi = height_vec[1] * 1000.0 # m
    TLo = T_vec[iLo]
    alphaTemp = 0 # lapse rate for current level
    # Exponential offset
    P_base = 101325 * np.exp(-MgR*zLo/TLo)
    # Loop through altitudes and compute pressure
    for iPoint in range(T_K.size):
        i_sort = sort_idx[iPoint]
        zCurr = z_m[i_sort]
        while zCurr > zHi: # march upwards and calculate pressure at the next layer boundary using barometric formula
            if np.abs(alphaTemp) > 0: # power law if there's a nonzero lapse rate
                PNew = P_base * np.power(T_vec[iHi]/T_vec[iLo],MgR/-alphaTemp)
            else: # exponential if lapse rate is zero (isothermal layer)
                PNew = P_base * np.exp(MgR*(zLo-zHi)/TLo)
            
            #fprintf('%5.2f km, %5.2f K, %9.2f -> %9.2f hPa, %8.5f K/m,\n',HVec(iLo),TVec(iLo),PBase./100,PNew./100,alphaTemp);
            P_base = PNew 
            iLo    = iHi
            iHi   += 1
            zLo    = zHi
            zHi    = height_vec[iHi] * 1000
            TLo    = T_vec[iLo]
            alphaTemp = lapse_vec[iLo] / 1000
            
        # calculate pressure at zCurr, using the last known P_base at zLo down/up to zCurr
        # pressures align back to the original unsorted altitude order
        if np.abs(alphaTemp) > 0:
            p_pa[i_sort] = P_base * np.power(T_K[i_sort]/TLo,MgR/-alphaTemp)
        else:
            p_pa[i_sort] = P_base * np.exp(MgR*(zLo-zCurr)/TLo)

    # Also calculate air density in kg/m3
    rho_kgm3 = (28.97e-3) * p_pa[sort_idx] / (8.314 * T_K[sort_idx])
    #dynVisc = (np.power(T_K,1.5) * 1.458e-6)/(T_K + 110.4)
    #kinVisc = dynVisc/rho_kgm3

    return p_pa, T_K, rho_kgm3

def altitude_to_pressure(z_m):
    p_pa, T_K, rho_kgm3 = altitude_to_many(z_m)
    return p_pa

def pressure_to_altitude(p_pa):
    '''ATMOSPALT Lightweight version of atmospalt from the MATLAB Aerospace Toolbox
    Returns the COESA estimate of altitude (in m) for a pressure (in Pa)'''
    z_int = None
    p_int = None

    # Generate interpolation vecor
    if z_int is None or p_int is None:
        z_int = np.arange(82e3,-1e3,-1.0)
        p_int = altitude_to_pressure(z_int)

    z_m = np.interp(np.array(p_pa),p_int,z_int,np.inf,z_int[-1]) # includes out of range values
    return z_m

def convert_pressure_to_altitude(df):
    '''
    Takes dataframe of planeflight.log output and adds altitude column

    Args:
        df: DataFrame with 'PRESS_bin' column in hPa, from load_planeflight function
    
    Returns:
        df: DataFrame with 'altitude_km' column added, in km
    
    '''
    p_pa = df['PRESS_bin'] * 100
    z_m = pressure_to_altitude(p_pa.values)
    df['altitude_km'] = z_m / 1000

    return df

### Unit conversions
def molec_cm3_to_vv(conc_molec_cm3, alt_km, output):
    '''
    Convert molec/cm3 to ppbv or pptv at a given altitude

    Args:
        conc_molec_cm3: concentration in molec/cm3
        alt_km: altitude in km
        output: 'ppbv' or 'pptv'
    Returns:
        concentration in ppbv or pptv
    
    '''
    R_star = 8.3144598 # universal gas constant J/K/mol
    avogadro = 6.023e23 

    P_Pa, T_K, _ = altitude_to_many(np.asarray(alt_km) * 1000)
    
    # compute number density of air (molec/cm3)
    n_air_cm3 = (avogadro * P_Pa) / (R_star * T_K) * 1e-6

    if output == 'ppbv':
        conversion = 1e9
    elif output == 'pptv':
        conversion = 1e12
    
    return (np.asarray(conc_molec_cm3) / n_air_cm3) * conversion


def vv_to_molec_cm3(vv, alt_km, input_units):
    '''
    Convert ppbv or pptv to molec/cm3 at a given altitude

    Args:
        vv: mixing ratio in ppbv or pptv
        alt_km: altitude in km
        unit: 'ppbv' or 'pptv'

    Returns:
        concentration in molec/cm3
    '''
    R_star    = 8.3144598  # universal gas constant J/K/mol
    avogadro  = 6.023e23  

    P_Pa, T_K, _ = altitude_to_many(np.asarray(alt_km) * 1e3)

    # compute number density of air (molec/cm3)
    n_air_cm3 = (avogadro * P_Pa) / (R_star * T_K) * 1e-6

    if input_units == 'ppbv':
        conversion = 1e9
    elif input_units == 'pptv':
        conversion = 1e12

    # conc = vv_fraction * n_air
    return (np.asarray(vv) / conversion) * n_air_cm3

# function to calculate k_CH4+OH given temperature in K
def calc_k(T):
    '''
    Calculate the reaction rate constant k for CH4 + OH given temperature in K.
    Constants are from rateLawUtilFuncs in KPP/fullchem
    '''
    A0 = 2.45e-12
    C0 = -1775

    return A0 * np.exp(C0 / T)
