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
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import ListedColormap
import datetime
import h5py
from pyproj import Geod
from shapely.geometry import Point, LineString, Polygon
import math
import scipy.stats as st
from scipy.interpolate import interp1d
import copy
import calendar
import time
from joblib import Parallel, delayed

from utils import standard_value, get_pressure_tropomi, convert_bytes_to_datetimes, get_valid_tropomi, regrid_sw_vertically, cal_avg, save_output_grid_variables_monthly

'''
The following code reads S5P L2 data for NO2 and processes the retrievals for comparison with GEOS-Chem. Modified from Ruijun Dang.
'''

def read_tropomi_l2(tropomi1,tropomi2):
    
    factor = 6.022141e19   # unit conversion, multiplication factor to convert mol/m2 (original unit) to molec/cm2
    scd = standard_value(tropomi2['DETAILED_RESULTS'],'nitrogendioxide_slant_column_density')[0,:,:]*factor                          #2d  unit: molec cm-2  [4172, 450]
    scd_uncertainty = standard_value(tropomi2['DETAILED_RESULTS'],'nitrogendioxide_slant_column_density_precision')[0,:,:]*factor    #2d  unit: molec cm-2
    vcd = standard_value(tropomi2['DETAILED_RESULTS'],'nitrogendioxide_total_column')[0,:,:]*factor                                  #2d  unit: molec cm-2
    vcd_uncertainty = standard_value(tropomi2['DETAILED_RESULTS'],'nitrogendioxide_total_column_precision')[0,:,:]*factor            #2d  unit: molec cm-2
    vcd_strat = standard_value(tropomi2['DETAILED_RESULTS'],'nitrogendioxide_stratospheric_column')[0,:,:]*factor                    #2d  unit: molec cm-2
    vcd_strat_uncertainty = standard_value(tropomi2['DETAILED_RESULTS'],'nitrogendioxide_stratospheric_column_precision')[0,:,:]*factor
    vcd_trop = standard_value(tropomi1,'nitrogendioxide_tropospheric_column')[0,:,:]*factor                                          #2d  unit: molec cm-2
    vcd_trop_uncertainty = standard_value(tropomi1,'nitrogendioxide_tropospheric_column_precision')[0,:,:]*factor                    #2d  unit: molec cm-2
    
    amf = standard_value(tropomi1,'air_mass_factor_total')[0,:,:]                 #2d  unitless [scanline(changable),ground_pixel(450)]
    amf_trop = standard_value(tropomi1,'air_mass_factor_troposphere')[0,:,:]      #2d  unitless [scanline(changable),ground_pixel(450)]
    ak = standard_value(tropomi1,'averaging_kernel')[0,:,:]                       #3d  unitless [scanline(changable),ground_pixel(450),layer(34)]
    
    lat = standard_value(tropomi1,'latitude')[0,:,:]                                      #2d  unit: degree [scanline(changable),ground_pixel(450)]
    lon = standard_value(tropomi1,'longitude')[0,:,:]                                     #2d  unit: degree center of the groundpixel
    lat_corners = standard_value(tropomi2['GEOLOCATIONS'],'latitude_bounds')[0,:,:]       #2d  unit: degree [scanline(changable),ground_pixel(450),corner(4)]
    lon_corners = standard_value(tropomi2['GEOLOCATIONS'],'longitude_bounds')[0,:,:]      #2d  unit: degree corner of the groundpixel
    utc_time = convert_bytes_to_datetimes(tropomi1['time_utc'][:])
    utc_time = np.repeat(utc_time,scd.shape[1],0)
    utc_time = utc_time.T
    sza = standard_value(tropomi2['GEOLOCATIONS'],'solar_zenith_angle')[0,:,:]            #2d  unit: degree
    vza = standard_value(tropomi2['GEOLOCATIONS'],'viewing_zenith_angle')[0,:,:]          #2d  unit: degree
    surf_pres = standard_value(tropomi2['INPUT_DATA'],'surface_pressure')[0,:,:]/100      #2d  unit: hPa [scanline(changable),ground_pixel(450)]
    eta_a = standard_value(tropomi1,'tm5_constant_a')/100                          #2d  unit: hPa [layer(34),vertices(2)]
    eta_b = standard_value(tropomi1,'tm5_constant_b')                              #2d  unit: 1 [layer(34),vertices(2)]
    lower_pressure = get_pressure_tropomi(surf_pres,eta_a[:,0],eta_b[:,0])         #3d  unit: hPa [scanline(changable),ground_pixel(450),layer(34)]
    upper_pressure = get_pressure_tropomi(surf_pres,eta_a[:,1],eta_b[:,1])         #3d  unit: hPa [scanline(changable),ground_pixel(450),layer(34)]
    pressure = (lower_pressure + upper_pressure)/2                                 #3d  unit: hPa [scanline(changable),ground_pixel(450),layer(34)]
    troppause_index = standard_value(tropomi1,'tm5_tropopause_layer_index')[0,:,:]        #2d  unit: 1 [scanline(changable),ground_pixel(450)]
    # layer index starts from 0, and this trop_pause_index is the highest layer in the troposphere
    
    cld_frac = standard_value(tropomi2['DETAILED_RESULTS'],'cloud_fraction_crb_nitrogendioxide_window')[0,:,:]     #2d  unitless
    cld_pres = standard_value(tropomi2['DETAILED_RESULTS']['FRESCO'],'fresco_cloud_pressure_crb')[0,:,:]/100       #2d  unit: hPa
    cld_pres_o22 = standard_value(tropomi2['DETAILED_RESULTS']['O22CLD'],'o22cld_cloud_pressure_crb')[0,:,:]/100   #2d  unit: hPa
    cld_radi_frac = standard_value(tropomi2['DETAILED_RESULTS'],'cloud_radiance_fraction_nitrogendioxide_window')[0,:,:]      #2d  unitless
    cld_scene_pres = standard_value(tropomi2['DETAILED_RESULTS']['FRESCO'],'fresco_apparent_scene_pressure')[0,:,:]/100       #2d  unit: hPa
    cld_scene_pres_o22 = standard_value(tropomi2['DETAILED_RESULTS']['O22CLD'],'o22cld_apparent_scene_pressure')[0,:,:]/100   #2d  unit: hPa
    albedo = standard_value(tropomi2['INPUT_DATA'],'surface_albedo_nitrogendioxide_window')[0,:,:]                 #2d  unitless
    main_flag = standard_value(tropomi1,'qa_value')[0,:,:]*0.01                                                    #2d  [0,1] the larger, the better
    snw_flag = standard_value(tropomi2['INPUT_DATA'],'snow_ice_flag')[0,:,:]                                       #2d  unitless, 0: no snow/ice, different value indicate different status
    
    
    data = {'scd':scd,'vcd':vcd,
            'vcd_strat':vcd_strat,'vcd_trop':vcd_trop,
            'amf_trop':amf_trop,'amf':amf,'ak':ak,
            'lat':lat,'lon':lon,'utc_time':utc_time,
            'latcorners':lat_corners,'loncorners':lon_corners,
            'pressure':pressure,'troppause_index':troppause_index,
           }
    error= {'scd':scd_uncertainty,
            'vcd':vcd_uncertainty,
            'vcd_strat':vcd_strat_uncertainty,
            'vcd_trop':vcd_trop_uncertainty
           }
    flag = {'cld_frac':cld_frac,'cld_pres':cld_pres,'cld_pres_o22':cld_pres_o22,
            'cld_rfrac':cld_radi_frac,'cld_scene_pres':cld_scene_pres,'cld_scene_pres_o22':cld_scene_pres_o22,
            'sza':sza,'vza':vza,'albedo':albedo,
            'main_flag':main_flag,'snw_flag':snw_flag,
           }
    return data,error,flag

def get_gc_result(file1, file2):
    gc = xr.open_dataset(file1)
    gc_no2_pptv = gc['SatDiagnConc_NO2'] * 1e12                           # unit: pptv, shape: 31 x 47 x 91 x 144 
    gc_no2_nd = gc['SatDiagnConc_NO2'] * gc['SatDiagnAirDen']             # unit: molec cm-3
    gc_no2_vcdl = gc_no2_nd * gc['SatDiagnBoxHeight']*100                 # unit: molec cm-2
    troppause = gc['SatDiagnTROPP']                                       # unit: hPa
    pressure_edges = xr.open_dataset(file2)['SatDiagnPEDGE']
    pressure = (pressure_edges[:,:47,:,:].drop('ilev') + pressure_edges[:,1:,:,:].drop('ilev'))/2
    pressure = pressure.rename({'ilev': 'lev'})
    # pressure = gc['SatDiagnPEdge']
    data = xr.Dataset({
        'no2_pptv': gc_no2_pptv,
        'no2_nd': gc_no2_nd,
        'no2_vcdl': gc_no2_vcdl,
        'pressure': pressure,
        'troppause':troppause,
    })
    return data

def create_output_grid_variables_monthly(gridtimes, gridlats, gridlons):
    shape_3d = (len(gridtimes), len(gridlats), len(gridlons))
    variables = {
        'num': np.zeros(shape_3d),                    # number of all valid observations

        # original tropomi data
        'VCD_tropomi': np.zeros(shape_3d),            # total vertical column density
        'VCDtrop_tropomi': np.zeros(shape_3d),        # the trop part
        'VCDstrat_tropomi': np.zeros(shape_3d),       # the strat part
        
        # recalculated tropomi with gc profile
        'VCD_tropomi_gc': np.zeros(shape_3d),         # total vertical column density, using GC profile
        'VCDtrop_tropomi_gc': np.zeros(shape_3d),     # the trop part, using GC profile
        
        # original geos-chem results
        'VCD_gc': np.zeros(shape_3d),                 # total vertical column density
        'VCDtrop_gc': np.zeros(shape_3d),             # the trop part
        
        # gc with ak applied
        'VCD_gc_ak': np.zeros(shape_3d),                 # total vertical column density
        'VCDtrop_gc_ak': np.zeros(shape_3d),             # the trop part
                
    }
    return variables

def process_tropomi_file(file0):
    
    ### 1 ########################################
    ### read TROPOMI NO2 data
    file_path = os.path.join(tropomi_path + yyyymm, file0)
    f0 = h5py.File(file_path, mode='r')
    product = f0['PRODUCT']
    support = f0['PRODUCT']['SUPPORT_DATA']
    pxTROPOMI_data, pxTROPOMI_error, pxTROPOMI_flg = read_tropomi_l2(product, support)
    f0.close()

    ### 2 ########################################  
    ### data filtering, get valid TROPOMI data
    # Remove grid cells with 
    # (0) quality flag < 0.75
    # (1) surface albedo larger than 0.3 
    # (2) with sza larger than 70 
    # (3) with snow_flag!=0
    # (we need snow/ice free data, snow_flag==0 indicates snow over land, ==255 indicates snow over ocean)
    mask_tropomi = (pxTROPOMI_flg['main_flag']>0.75) & \
                    (pxTROPOMI_flg['albedo']<0.3) & \
                    ((pxTROPOMI_flg['snw_flag']==0) | (pxTROPOMI_flg['snw_flag']==255)) & \
                    (pxTROPOMI_flg['sza']<70)
    if ((mask_tropomi).sum()==0):
        print('Skipping TROPOMI (no valid obs):',file0[20:])
        return None
    print('Processing TROPOMI:',file0[20:])
    # apply the mask to the TROPOMI to obtain valid data
    # the shape of TROPOMI data are no longer in 2d, is now in 1d
    pxTROPOMI_data,pxTROPOMI_error,pxTROPOMI_flg = \
                    get_valid_tropomi(pxTROPOMI_data,pxTROPOMI_error,pxTROPOMI_flg,mask_tropomi)
    
    partial_output = create_output_grid_variables_monthly(gridtime, gridlat, gridlon)
    
    ### 3 ########################################
    ### mapping TROPOMI pixel to 2x2.5 gridcells
    # 获取 TROPOMI 点
    lats = pxTROPOMI_data['lat']
    lons = pxTROPOMI_data['lon']
    # 找到每个点属于哪个格子（返回的是索引）
    lats_idx = np.digitize(lats, gridlat_edges) - 1
    lons_idx = np.digitize(lons, gridlon_edges) - 1
    # 快速获取 unique grid pair + inverse map
    grids = np.stack([lats_idx, lons_idx], axis=1)
    unique_grids, inverse = np.unique(grids, axis=0, return_inverse=True)

    ### 4 ########################################
    ### find geos-chem data on the same day
    date = pxTROPOMI_data['utc_time'][0].strftime('%Y-%m-%d %H:%M:%S')[:10]
    # in case last file is already in next month
    if date > str(gridtime[-1])[:10]:
        return None
    gridGC_data = gridGC_monthly.sel(time=date)
    day_idx = int(date[-2:])-1

    # TEST
    # if gridGC_data['pressure'] is all -999, print the date and skip the file
    if np.all(gridGC_data['pressure'].values == -999):
        print('-999 pressure, skip ',file0[20:], ' date:', date)
        return None

    ### 5 ########################################
    ### iterate over all unique_grids of TROPOMI
    for g in range(len(unique_grids)):
        
        lat_idx,lon_idx = unique_grids[g]
        # deal with boundaries
        # when it is closest to the +180, meaning it is also closest to -180
        lon_idx = np.mod(lon_idx,len(gridlon))
        if lat_idx == len(gridlat):
            lat_idx = lat_idx-1
        pixel_idx = np.arange(len(lats))[inverse==g]
            
        # get GEOS-Chem results for this grid cell
        gridPressure = gridGC_data.isel(lon=lon_idx,lat=lat_idx)['pressure'].values
        gridTPause   = gridGC_data.isel(lon=lon_idx,lat=lat_idx)['troppause'].values
        gridVCDL     = gridGC_data.isel(lon=lon_idx,lat=lat_idx)['no2_vcdl'].values

        # iterate over all pixels within this grid cell
        for p in pixel_idx:
            
            #### data for this TROPOMI pixel
            pxPressure = pxTROPOMI_data['pressure'][p]
            pxTPausein = int(pxTROPOMI_data['troppause_index'][p])
            pxTPause   = pxPressure[pxTPausein:pxTPausein+2].mean()
            pxAK       = pxTROPOMI_data['ak'][p]
            pxAMF      = pxTROPOMI_data['amf'][p]
            pxAMFt     = pxTROPOMI_data['amf_trop'][p]
            pxVCD     = pxTROPOMI_data['vcd'][p]
            pxVCDt    = pxTROPOMI_data['vcd_trop'][p]
            pxVCDs    = pxTROPOMI_data['vcd_strat'][p]
            
            ###### For TROPOMI vs GC comparison
            ###### Approach 1 and 2 are theoretically equivalent
            ###### they should lead to the same NMB on pixel level
            
            ####------Approach1: recalculate TROPOMI vcd using GEOS-Chem profile
            # interpolate TROPOMI scattering weight on GC pressure level
            pxSW = pxAMF * pxAK
            pxSW_GC = regrid_sw_vertically(pxSW,pxPressure,gridPressure)
            # get troposphere mask
            pxTrop_GC = gridPressure > pxTPause        # use TROPOMI tropopause
            # pxTrop_GC = gridPressure > gridTPause    # use GC tropopause
            # recalculate AMF using GEOS-Chem NO2 profile (VCDL)
            pxAMF_GC = sum(pxSW_GC*gridVCDL)/sum(gridVCDL)
            pxAMFt_GC = sum(pxSW_GC[pxTrop_GC]*gridVCDL[pxTrop_GC])/sum(gridVCDL[pxTrop_GC])
            # Using  AMF_new to recalculate TROPOMI_VCD=(VCD_old*AMF_old/AMF_new)
            pxVCD_GC = pxVCD   * pxAMF  / pxAMF_GC
            pxVCDt_GC = pxVCDt * pxAMFt / pxAMFt_GC
            # get GEOS-Chem simulated trop. column using TROPOMI tropopause
            gridVCD  = sum(gridVCDL)
            gridVCDt = sum(gridVCDL[pxTrop_GC])
            
            ### assign TROPOMI_orignal, TROPOMI_new and GEOS-Chem to gridded output
            partial_output['num'][day_idx,lat_idx,lon_idx] += 1
            partial_output['VCD_tropomi'][day_idx,lat_idx,lon_idx] += pxVCD
            partial_output['VCDtrop_tropomi'][day_idx,lat_idx,lon_idx] += pxVCDt
            partial_output['VCDstrat_tropomi'][day_idx,lat_idx,lon_idx] += pxVCDs
            partial_output['VCD_tropomi_gc'][day_idx,lat_idx,lon_idx] += pxVCD_GC
            partial_output['VCDtrop_tropomi_gc'][day_idx,lat_idx,lon_idx] += pxVCDt_GC
            partial_output['VCD_gc'][day_idx,lat_idx,lon_idx] += gridVCD
            partial_output['VCDtrop_gc'][day_idx,lat_idx,lon_idx] += gridVCDt
            
            ####------Approach2: applying TROPOMI averaging kernel to GEOS-Chem
            # interpolate TROPOMI averaging kernel on GC pressure level
            pxAK_GC = regrid_sw_vertically(pxAK,pxPressure,gridPressure)
            # get troposphere mask
            pxTrop_GC = gridPressure > pxTPause        # use TROPOMI tropopause
            # applying ak to GEOS-Chem results
            gridVCD_ak = sum(gridVCDL*pxAK_GC)
            gridVCDt_ak = sum(gridVCDL[pxTrop_GC]*pxAK_GC[pxTrop_GC]*(pxAMF/pxAMFt)) # converts total column AK to tropospheric AK
            ### assign GEOS-Chem_ak to gridded output
            partial_output['VCD_gc_ak'][day_idx,lat_idx,lon_idx] += gridVCD_ak
            partial_output['VCDtrop_gc_ak'][day_idx,lat_idx,lon_idx] += gridVCDt_ak

    return partial_output


#######--------parallel processing--------########
tropomi_months = pd.date_range(start='2024-01-01', end='2024-12-01', freq='MS').strftime('%Y%m').tolist() 

res = '4x5'
exp = 'gc_4x5_merra2_14.7/standard'
explabel = 'standard_v2'

tropomi_path_RJ = '/n/holylfs05/LABS/jacob_lab/Users/rdang/Obs-data/TROPOMI/NO2/' # Ruijun's, 202401-202409
tropomi_path_MH = '/n/holylfs05/LABS/jacob_lab/Users/mhe/Obs_data/TROPOMI/NO2/' # mine, 202410-202412
geoschem_path = '/n/holylfs06/LABS/jacob_lab2/Lab/mhe/GlobalOH/'+exp+'/OutputDir/'
output_path = '/n/holylfs05/LABS/jacob_lab/Users/mhe/Obs_data/TROPOMI_regrid/NO2/'

for yyyymm in tropomi_months:
    print('date: ', yyyymm)
    # if month is 10,11,12, use MY tropomi path
    if yyyymm[4:] in ['10','11','12']:
        tropomi_path = tropomi_path_MH
    else: # use Ruijun's path
        tropomi_path = tropomi_path_RJ
    outputfile = output_path+'tropomi_with_gc_'+explabel+'_'+res+'_'+yyyymm+'.nc'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

     # get TROPOMI files
    file_list = sorted(os.listdir(tropomi_path + yyyymm))
    # only keep files that end in .nc
    file_list = [f for f in file_list if f.endswith('.nc')]
    print(f'Number of TROPOMI files: {len(file_list)}')

    ### 0 ########################################
    ### prepare corresponding GEOS-Chem monthly data first
    gc_file1 = geoschem_path+'GEOSChem.SatDiagn.'+yyyymm+'01_0000z.nc4'
    gc_file2 = geoschem_path+'GEOSChem.SatDiagnEdge.'+yyyymm+'01_0000z.nc4'
    gridGC_monthly = get_gc_result(gc_file1, gc_file2)
    ### prepare the output datasets
    gridlat = gridGC_monthly.lat.values
    gridlon = gridGC_monthly.lon.values
    gridtime = gridGC_monthly.time.values
    gridlat_edges = gridlat - 1.0   
    gridlon_edges = gridlon - 1.25  
    gridlat_edges = np.append(gridlat_edges, 90)
    gridlon_edges = np.append(gridlon_edges, 180)
    gridlat_edges[0] = -90
    gridlon_edges[0] = -180

    # results = [process_tropomi_file(file) for file in file_list]
    results = Parallel(n_jobs=-1)(delayed(process_tropomi_file)(filename) for filename in file_list)
    print('len(results): ', len(results))
    results = [r for r in results if r is not None]
    print('len(results) after removing Nones: ', len(results))
    
    final_output = create_output_grid_variables_monthly(gridtime, gridlat, gridlon)

    for partial in results:
        for key in final_output:
            final_output[key] += partial[key]

    final = save_output_grid_variables_monthly(gridtime, gridlat, gridlon, final_output, outputfile)
    print(f"Wrote files to {output_path}\n")