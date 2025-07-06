import numpy as np
import xarray as xr
import os
import calendar
import warnings
warnings.filterwarnings("ignore")

'''
Compute methane lifetimes with respect to tropospheric OH loss, and column methane loss rates, for GEOS-Chem full-chemistry simulation.
Tropospheric OH loss is available as a specific output variable from the CH4 simulations, 
but not the full chemistry simulation (because it uses a unified strat-trop chemistry). 
The full chemistry simulation therefore needs to use a tropopause-based masking of its troposphere-stratosphere OH loss output.
Adapted from Todd Mooring (7/2/25)
'''

rundir = 'gc_4x5_merra2_fullchem'

# constants
AVOGADRO = 6.022140857e23  # molecules/mol
MOLAR_MASS_CH4 = 16.04      # g/mol
SECONDS_PER_YEAR = 3600 * 24 * 365.25
DRY_AIR_MOLAR_MASS = 28.9644  # g/mol
YEAR = 2016

# file paths
base = '/n/holylfs06/LABS/jacob_lab2/Lab/mhe/GlobalOH'
pattern = {
    'rxn_rates': base + f'/{rundir}/OutputDir/GEOSChem.RxnRates.{YEAR}{{month:02d}}01_0000z.nc4',
    'state_met': base + f'/{rundir}/OutputDir/GEOSChem.StateMet.{YEAR}{{month:02d}}01_0000z.nc4',
    'species_conc': base + f'/{rundir}/OutputDir/GEOSChem.SpeciesConc.{YEAR}{{month:02d}}01_0000z.nc4'
}

def get_month_weights(year=YEAR):
    days = np.array([calendar.monthrange(year, m)[1] for m in range(1, 13)], dtype=float)
    return days / days.sum()
    
def compute_ch4_mass():
    '''Compute global annual mean CH4 mass (kg) from mixing ratio and dry air mass'''
    ds0 = xr.open_dataset(pattern['species_conc'].format(month=1))
    nlat = ds0.dims['lat'] # 46
    nlon = ds0.dims['lon'] # 72
    nlev = ds0.dims['lev'] # 47

    ch4_mixr = np.full((nlon, nlat, nlev, 12), np.nan) # (72, 46, 47, 12)
    dry_mass = np.full((nlon, nlat, nlev, 12), np.nan) # (72, 46, 47, 12)

    for m in range(1, 13):
        # mixing ratio (mol CH4 / mol dry air)
        fn_ch4 = pattern['species_conc'].format(month=m)
        with xr.open_dataset(fn_ch4) as ds_ch4:
            ds_ch4_transpose = ds_ch4['SpeciesConcVV_CH4'].isel(time=0).transpose('lon', 'lat', 'lev').values
            ch4_mixr[..., m-1] = ds_ch4_transpose

        # dry air mass (kg)
        fn_met = pattern['state_met'].format(month=m)
        with xr.open_dataset(fn_met) as ds_met:
            ds_met_transpose = ds_met['Met_AD'].isel(time=0).transpose('lon', 'lat', 'lev').values
            dry_mass[..., m-1] = ds_met_transpose

    # convert dry mass: kg -> g -> mol dry air
    dry_mol = (dry_mass * 1000.0) / DRY_AIR_MOLAR_MASS
    # mol CH4 = mixing ratio * mol dry air
    ch4_mol = ch4_mixr * dry_mol # shape (lon, lat, lev, month)
    # sum over lon, lat, lev dims
    monthly_sum = ch4_mol.sum(axis=(0, 1, 2))  # length-12
    weights = get_month_weights()
    annual_mol = np.dot(weights, monthly_sum)
    # convert to mass: mol -> g -> kg
    annual_g = annual_mol * MOLAR_MASS_CH4
    annual_kg = annual_g * 1e-3  # kg
    return annual_kg

# Full chemistry OH loss calculation
def load_rxnrate_gridboxvol():
    '''Open CH4 + OH reaction rates (EQ025) and gridbox volumes for each month.'''

    # get grid sizes from 1st file
    ds0 = xr.open_dataset(pattern['rxn_rates'].format(month=1))
    nlat = ds0.dims['lat'] # 46
    nlon = ds0.dims['lon'] # 72
    nlev = ds0.dims['lev'] # 47

    rxnrate_025 = np.full((nlon, nlat, nlev, 12), np.nan) # (72, 46, 47, 12)
    gridbox_vol = np.full_like(rxnrate_025, np.nan)

    for m in range(1, 13):
        fn_rxn = pattern['rxn_rates'].format(month=m)
        fn_met = pattern['state_met'].format(month=m)
        with xr.open_dataset(fn_rxn) as ds_r:
            ds_r_transpose = ds_r['RxnRate_EQ025'].isel(time=0).transpose('lon', 'lat', 'lev').values
            rxnrate_025[..., m-1] = ds_r_transpose # molec/cm3/s (NOT grid-box integrated)
        with xr.open_dataset(fn_met) as ds_m:
            ds_m_transpose = ds_m['Met_AIRVOL'].isel(time=0).transpose('lon', 'lat', 'lev').values
            gridbox_vol[..., m-1] = ds_m_transpose # m3
    
    return rxnrate_025, gridbox_vol

def load_pressure_edges_and_tropopause():
    '''Load pressure edges and tropopause pressure for each month.
    Note: Met_PEDGE from LevelEdgeDiags might be more appropriate, but that would have required rerunning a simulation.'''
    
    path = base + f'/{rundir}/OutputDir/GEOSChem.StateMet.{YEAR}{{:02d}}01_0000z.nc4'
    # get grid sizes from 1st file
    ds0 = xr.open_dataset(path.format(1))
    nlat = ds0.dims['lat']
    nlon = ds0.dims['lon']
    nlev = ds0['hyai'].size

    # construct arrays
    ps = np.full((nlon, nlat, 12), np.nan)
    tropp = np.full((nlon, nlat, 12), np.nan)
    hyai, hybi = None, None

    for m in range(1, 13):
        ds = xr.open_dataset(path.format(m))
        p2d = ds['Met_PSC2WET']
        t2d = ds['Met_TropP']
        if 'time' in p2d.dims:
            p2d = p2d.isel(time=0)
            t2d = t2d.isel(time=0)
        arr_ps = p2d.values  # shape (lat, lon)
        arr_tp = t2d.values

        # transpose to (lon, lat)
        if arr_ps.shape == (nlat, nlon):
            arr_ps = arr_ps.T
            arr_tp = arr_tp.T

        ps[:, :, m-1]    = arr_ps
        tropp[:, :, m-1] = arr_tp

        # read edges
        if m == 1:
            hyai = ds['hyai'].values  # (lev+1,)
            hybi = ds['hybi'].values
        
    # build phalf
    # expand ps to (lon, lat, 1, month)
    ps4 = ps[:, :, None, :]

    # expand hyai, hybi to (1, 1, lev+1, 1)
    hyai4 = hyai[None, None, :, None]
    hybi4 = hybi[None, None, :, None]

    phalf = hyai4 + hybi4 * ps4
    
    return phalf, tropp

def mask_volumes(grid_vol, phalf, tropp):
    '''
    Mask real volumes to effective volumes.
    Zeroes out the parts of each model grid box volume that are above the tropopause, 
    so that when we later multiply the grid box volume by OH reaction rate, we only integrate over troposphere.
    Adjust gridbox volumes so that they correspond only to the (possibly zero) part of the gridbox that is in the troposphere.
    Plots of the lon-lat-month mean of gridbox_vol before and after masking is applied suggest that this works as expected.

    Args:
        grid_vol: 4d array of grid box volumes (lon x lat x lev x month)
        phalf: pressures at layer edges for each column (lon x lat x lev x month)
        tropp: tropopause pressure (hPa) for each column (lon x lat x month)
    '''
    # handle case where data is 1-month rather than annual
    if grid_vol.ndim == 3:
        grid_vol = grid_vol[..., None]
    if phalf.ndim == 3:       # (lon,lat,lev+1)
        phalf = phalf[..., None]
    if tropp.ndim == 2:       # (lon,lat)
        tropp = tropp[..., None]

    masked = grid_vol.copy() # lon x lat x lev x month
    nx, ny, nz, nt = masked.shape
    # loop over every column i,j and month t
    for i in range(nx):
        for j in range(ny):
            for t in range(nt):
                bot = phalf[i,j,:-1,t] # bottom pressure of each layer
                top = phalf[i,j,1:,t]
                trop = tropp[i,j,t]
                above = bot <= trop # the entire layer sits above the tropopause, so set masked to 0
                masked[i,j,above,t] = 0.0
                idx = np.where(~above)[0]
                if idx.size: 
                    k = idx[-1]
                    # find last layer that is below the tropopause. If tropopause pressure is within that layer, compute fraction of that layer's mass below the tropopause
                    if top[k] < trop < bot[k]:
                        frac = np.log(bot[k]/trop) / np.log(bot[k]/top[k])
                        masked[i,j,k,t] *= frac # cuts off the stratospheric part of the layer

    if masked.shape[-1] == 1:
        return masked[..., 0]
    return masked

def fullchem_loss_rate():
    rxnrate, vol = load_rxnrate_gridboxvol()
    phalf, tropp = load_pressure_edges_and_tropopause()
    # convert molec/cm3/s to mol/m3/s
    rxn = (rxnrate * 1e6) / AVOGADRO
    vol_trop = mask_volumes(vol, phalf, tropp)
    monthly = (rxn * vol_trop).sum(axis=(0,1,2)) # mol/s per box, grid-box integrated (lon, lat, lev, month)
    weights = get_month_weights()
    mean_mol_s = np.dot(weights, monthly)
    kg_s = mean_mol_s * MOLAR_MASS_CH4 * 1e-3 # kg/s per box

    return kg_s

def column_fullchem_loss_rate():
    rxnrate, vol = load_rxnrate_gridboxvol()
    phalf, tropp = load_pressure_edges_and_tropopause()
    # convert molec/cm3/s to mol/m3/s
    rxn = (rxnrate * 1e6) / AVOGADRO
    vol_trop = mask_volumes(vol, phalf, tropp)
    loss_box = rxn * vol_trop # (lon, lat, lev, month)

    # sum over levels to get column-integrated loss (mol/s per column)
    loss_col = loss_box.sum(axis=2)  # (lon, lat, month)

    # annual mean over months
    weights = get_month_weights()
    loss_col_annual = np.tensordot(loss_col, weights, axes=([2], [0]))  # (lon, lat)

    # convert mol/s to kg/s per column
    loss_col_kg_s = loss_col_annual * MOLAR_MASS_CH4 * 1e-3
    loss_col_kg_a = loss_col_kg_s * SECONDS_PER_YEAR

    # grab area
    ds0 = xr.open_dataset(pattern['rxn_rates'].format(month=1))
    area2d = ds0['AREA'].isel(lev=0) if 'lev' in ds0['AREA'].dims else ds0['AREA']
    area = area2d.values  # (lat, lon)
    lats = area2d['lat']
    lons = area2d['lon']

    # loss_kg_s is [lon,lat] -> transpose to [lat,lon]
    # divide by area to get kg/m2/a
    loss_col_m2 = loss_col_kg_a.T / area

    # create xarray DataArray with lats, lons, loss_col_m2
    loss_col_da = xr.DataArray(
        loss_col_m2,
        coords=[lats, lons],
        dims=['lat', 'lon'],
        name='column_methane_loss_rate',
        attrs={'units': 'kg/m2/a'}
    )

    return loss_col_da

def column_loss_month(month):
    '''
    Compute the column CH4 loss rate for a specific month.
    '''
    fn_rxn = pattern['rxn_rates'].format(month=month)
    fn_met = pattern['state_met'].format(month=month)
    with xr.open_dataset(fn_rxn) as ds_r, xr.open_dataset(fn_met) as ds_m:
        dr = ds_r['RxnRate_EQ025'].isel(time=0).transpose('lon','lat','lev')
        dv = ds_m['Met_AIRVOL'].isel(time=0).transpose('lon','lat','lev')
        area2d = ds_r['AREA'].isel(lev=0) if 'lev' in ds_r['AREA'].dims else ds_r['AREA']
        area = area2d.values  # (lat, lon)
        lats = area2d['lat']
        lons = area2d['lon']

    # convert molec/cm3/s to mol/m3/s
    dr.values = (dr.values * 1e6) / AVOGADRO
    
    # Mask above‐tropopause
    phalf, tropp = load_pressure_edges_and_tropopause()
    dv_trop = mask_volumes(dv.values, phalf[..., month-1], tropp[..., month-1])
    
    # integrate over column
    loss_mol_s = (dr.values * dv_trop).sum(axis=2)  # shape (lon,lat)
    
    # convert to kg/s and divide by area
    loss_col_kg_s = loss_mol_s * MOLAR_MASS_CH4 * 1e-3
    loss_col_kg_a = loss_col_kg_s * SECONDS_PER_YEAR
    loss_col_m2 = loss_col_kg_a.T / area

    loss_col_da = xr.DataArray(
        loss_col_m2,
        coords=[lats, lons],
        dims=['lat', 'lon'],
        name='column_methane_loss_rate',
        attrs={'units': 'kg/m2/a'}
    )

    return loss_col_da

def create_dataset():
    '''
    Create xarray dataset with monthly and annual column methane loss rates.
    '''
    # list of column loss rates for each month
    monthly = [column_loss_month(m) for m in range(1,13)]

    # concatenate
    loss_monthly = xr.concat(
        monthly,
        dim='month'
    )
    loss_monthly = loss_monthly.assign_coords(month=('month', np.arange(1,13)))
    loss_monthly.name = 'ch4_loss_monthly'
    loss_monthly.attrs['units'] = monthly[0].attrs.get('units', 'kg/m2/a')
    
    # compute the annual mean column loss
    loss_annual = column_fullchem_loss_rate()
    loss_annual.name = 'ch4_loss_annual'
    loss_annual.attrs['units'] = 'kg/m2/a'
    
    # merge
    ds = xr.Dataset({
        'loss_monthly': loss_monthly,
        'loss_annual':   loss_annual
    })
    return ds


if __name__ == '__main__':
    ch4_mass = compute_ch4_mass()
    loss = fullchem_loss_rate()
    lifetime_s = ch4_mass / loss
    lifetime_yr = lifetime_s / SECONDS_PER_YEAR
    print(f"Fullchem_4x5 CH4 mass: {ch4_mass:.3e} kg")
    print(f"Fullchem_4x5 CH4 lifetime: {lifetime_yr:.3f} years")
    print(f"Fullchem_4x5 CH4 tropospheric OH loss rate: {loss:.3e} kg/s")

    ds = create_dataset()
    
    if os.path.exists(f'{base}/CH4_loss.nc'):
        os.remove(f'{base}/CH4_loss.nc')
    ds.to_netcdf(f'{base}/CH4_loss.nc')