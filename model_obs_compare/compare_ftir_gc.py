from pyhdf.SD import SD, SDC
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import xarray as xr
import os
import re
import glob
from scipy.interpolate import interp1d

# Define scale height and surface pressure
H = 7.5  # km
P0 = 1013.25  # hPa

def get_site_name(ftir_path):
    filename = os.path.basename(ftir_path)
    match = re.search(
        r"ftir\.[^_]+_(.+?)_\d{8}t\d{6}z",
        filename
    )
    site_raw = match.group(1)
    site_clean = site_raw.replace('.', ' ').replace('_', ' ').title()
    return site_clean

def interp_to_ftir(gc_profile, p_gc, p_ftir):
    f = interp1d(np.log(p_gc), gc_profile,
                 bounds_error=False, fill_value="extrapolate")
    return f(np.log(p_ftir))


def read_gc(gc_path):
    # Load species conc
    speciesconc_ds = xr.open_mfdataset(gc_path + 'GEOSChem.SpeciesConc.2024*')
    # Load met
    met_ds = xr.open_mfdataset(gc_path + 'GEOSChem.StateMet.2024*')
    # Calculate pressure at midpoints of model layers
    p_mid_gc = (met_ds['hyam'] + met_ds['hybm'] * met_ds['Met_PSC2DRY']).mean(dim='time') # (lev=47, lat, lon)

    gc = speciesconc_ds['SpeciesConcVV_CH2O']

    return gc, p_mid_gc


########### Set paths ############
ftir_base = '/n/home12/mhe/lfs/Obs_data/FTIR/'
exp = 'standard'
gc_base = f'/n/holylfs06/LABS/jacob_lab2/Lab/mhe/GlobalOH/gc_4x5_merra2_14.7/{exp}/OutputDir/'

# Collect all FTIR H2CO files
ftir_files = sorted(glob.glob(os.path.join(ftir_base, "ftir.h2co_*.hdf")))
nfiles = len(ftir_files)
print(f'Found {nfiles} FTIR files.')

fig, axes = plt.subplots(1, nfiles, figsize=(4*nfiles, 6), sharey=True)

# Load GC output
gc, p_mid = read_gc(gc_base)

for i, ftir_path in enumerate(ftir_files):

    site = get_site_name(ftir_path)
    print(f'\nProcessing site: {site}')

    hdf = SD(ftir_path, SDC.READ)

    # check units
    sds = hdf.select('H2CO.MIXING.RATIO.VOLUME_ABSORPTION.SOLAR')

    obs_h2co = hdf.select('H2CO.MIXING.RATIO.VOLUME_ABSORPTION.SOLAR').get()
    obs_ak = hdf.select('H2CO.MIXING.RATIO.VOLUME_ABSORPTION.SOLAR_AVK').get()
    obs_apriori = hdf.select('H2CO.MIXING.RATIO.VOLUME_ABSORPTION.SOLAR_APRIORI').get()
    
    # check units
    # If ppmv, we need to convert to ppbv by multiplying by 1e3.
    if sds.attributes()['VAR_UNITS'] == 'ppbv':
        print('This site has units of ' + sds.attributes()['VAR_UNITS'] + ' instead of ppmv.')
    elif sds.attributes()['VAR_UNITS'] == 'ppmv':
        obs_h2co *= 1e3 #ppmv -> ppbv
        obs_apriori *= 1e3 #ppmv -> ppbv    
    else:
        raise ValueError(f"Unexpected units: {sds.attributes()['VAR_UNITS']}")

    alt  = hdf.select('ALTITUDE').get()
    if len(alt.shape) == 2:
        alt = alt[0]
    date = hdf.select('DATETIME').get()
    # convert time
    dates = [dt.datetime(2000,1,1) + dt.timedelta(days=float(d)) for d in date]
    print(f'Range of dates in the measurement: {dates[0]}, {dates[-1]}')

    months = np.array([d.month for d in dates])
    # equally weigh each available month in obs
    unique_months = np.unique(months)

    monthly_means_h2co = []
    monthly_means_apriori = []
    monthly_means_ak = []

    for m in unique_months:
        mask = months == m
        
        # Mean over time within this month
        monthly_means_h2co.append(np.nanmean(obs_h2co[mask, :], axis=0))
        monthly_means_apriori.append(np.nanmean(obs_apriori[mask, :], axis=0))
        monthly_means_ak.append(np.nanmean(obs_ak[mask, :, :], axis=0))

    monthly_means_h2co = np.array(monthly_means_h2co)
    monthly_means_apriori = np.array(monthly_means_apriori)
    monthly_means_ak = np.array(monthly_means_ak)

    # mean over sampling time
    obs_h2co_mean = np.nanmean(monthly_means_h2co, axis=0) # ppbv
    obs_apriori_mean = np.nanmean(monthly_means_apriori, axis=0) # ppbv
    obs_ak_mean = np.nanmean(monthly_means_ak, axis=0)

    z = alt

    site_lat =  hdf.select('LATITUDE.INSTRUMENT').get()[0]
    site_lon = hdf.select('LONGITUDE.INSTRUMENT').get()[0]
    print(f'site lat: {site_lat:.2f}, site lon: {site_lon:.2f}')

    # Find nearest lat/lon in GC
    gc_site = gc.sel(lat=site_lat, lon=site_lon, method='nearest') # VMR (v/v)
    
    # select only months that were in FTIR data
    gc_months = gc_site['time'].dt.month
    mask_gc = gc_months.isin(unique_months)
    gc_site_subset = gc_site.sel(time=mask_gc)

    gc_mean = gc_site_subset.mean('time').squeeze()

    p_gc_site = p_mid.sel(lat=site_lat, lon=site_lon, method='nearest').squeeze() # 47 levels, hPa

    alt_gc_site = -H * np.log(p_gc_site / P0) # km
    alt_gc_site = alt_gc_site.squeeze()

    print(f'GC site lat: {gc_site.lat.values}, GC site lon: {gc_site.lon.values}')

    # make sure pressure is 1d
    p_gc = p_gc_site.values
    x_gc = gc_mean.values * 1e9  # ppb

    p_ftir = np.nanmean(hdf.select('PRESSURE_INDEPENDENT').get(), axis=0)
    x_gc_interp = interp_to_ftir(x_gc, p_gc, p_ftir)

    # Smoothing (Rodgers)
    x_gc_smoothed = obs_apriori_mean + obs_ak_mean @ (x_gc_interp - obs_apriori_mean)

    # Plot
    ax = axes[i]

    ax.plot(obs_h2co_mean, z, label='FTIR', lw=3, color='black')
    ax.plot(x_gc_interp, z, '--', label='GC Raw', lw=2, color='orange')
    ax.plot(x_gc_smoothed, z, label='GC Smoothed', lw=3, color='red')

    ax.set_title(f'{site} ({site_lat:.1f}, {site_lon:.1f})')
    ax.set_xlabel("HCHO VMR (ppbv)")
    ax.set_xlim(0, None)
    ax.set_ylim(0, 15)
    ax.grid()

    if i == 0:
        ax.set_ylabel("Altitude (km)")
        ax.legend()

plt.tight_layout()
plt.savefig('ftir_gc_comparison.png', dpi=300)