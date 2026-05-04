from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd

from metpy.calc import dewpoint_from_specific_humidity
from metpy.units import units

import xesmf as xe

import os
import sys

sys.path.append('/home/548/cd3022/repos/solar-nowcast/modules')
import sat_preprocess
import clear_sky


date = sys.argv[1]
num_days = sys.argv[2]

start_year, start_month, start_day = [int(i) for i in date.split('-')]

# Data boundaries
start_date = datetime(start_year, start_month, start_day)
end_date   = datetime(start_year, start_month, start_day + int(num_days))

lat_min=-35
lat_max=-28.5
lon_min=145
lon_max=151.5


####################################################################
# HELIOSAT DATA
####################################################################

# Get file paths for Heliosat datasets
base_path = Path('/g/data/rv74/satellite-products/arc/der/himawari-ahi/solar/p1s/latest')

files = []

current = start_date
while current <= end_date:
    year  = current.year
    month = current.month
    day   = current.day

    file_path = base_path / f"{year}/{month:02d}/{day:02d}"
    
    if file_path.exists():  # important for missing days
        files.extend(file_path.rglob("*.nc"))
    
    current += timedelta(days=1)


# Select variables to be used from the heliosat dataset
helio_vars = [
    'surface_global_irradiance',
    'cloud_optical_depth',
    'solar_elevation',
]

# keep just the region and variables during preprocessing
def preprocess(ds):
    return ds.sel(
        latitude=slice(lat_min, lat_max),
        longitude=slice(lon_min, lon_max)
    )[helio_vars]


# Open the data
ghi = xr.open_mfdataset(files, preprocess=preprocess)


####################################################################
# RADIANCE DATA
####################################################################

rad_list = []

ch_list = [
    'B03',
    # 'B04',
    'B05',
    'B07',
    # 'B08',
    'B09',
    'B11',
    'B13',
    'B15'
]


for channel in ch_list:

    ds = sat_preprocess.read_himawari_channel(
        channel,
        start_date, end_date,
        lat_min, lat_max, lon_min, lon_max,
        coords=True
    )

    rad_list.append(ds)


# Use final radiance ds as reference for interpolation
# (if channels were listed in ascending order, should be the lowest resolution channel)
ref_ds = rad_list[-1]

interp_list = []

# interpolate radiance data to the same grid
for ds in rad_list:
    if ds.sizes != ref_ds.sizes:
        ds = ds.interp(y=ref_ds.y, x=ref_ds.x)

    # Drop conflicting coords (except for reference)
    if ds is not ref_ds:
        ds = ds.drop_vars(["latitude", "longitude"], errors="ignore")

    interp_list.append(ds)
    
# merge into single dataset
ds_rad = xr.merge(interp_list)

# Regrid radiances from (x, y) to (lat, lon)
regridder = xe.Regridder(
    ds_rad,
    ghi,
    method="bilinear",
    reuse_weights=False
)
ds_rad_interp = regridder(ds_rad)


####################################################################
# BARRA-R2 DATA
####################################################################
files = []
year  = start_date.year
month = start_date.month

# STANDARD VARS
variables_of_interest = [
    # Moisture
    'huss',
    'hus850',
    'hus700',
    'hus500',
    # Wind
    # 'ua850',
    # 'va850',
    # 'wa850',
    # Pressure and geopotential
    'psl',
    # 'zg850',
    # 'zg500',
    # Temperature
    'tas',
    'ta850',
    'ta700',
    'ta500',
]

# Add files to list for open_mfdataset
for var in variables_of_interest:
    file_path = Path(f'/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{var}/latest/')
    var_file = [f for f in file_path.glob(f'*{year}{month:02d}.nc')][0]
    files.append(var_file)

# CONVECTIVE VARS
variables_of_interest = [
    'RH24mean',
    'MUEL',
    'FZL',
    'MULCL'
]

# Add files to list for open_mfdataset
for var in variables_of_interest:
    file_path = Path(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/{var}/latest/')
    var_file = [f for f in file_path.glob(f'*{year}{month:02d}.nc')][0]
    files.append(var_file)

# Keep just time and region 
def preprocess(ds):
    return ds.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max),
        time=slice(np.datetime64(start_date), np.datetime64(end_date))
    )
# Open BARRA Data
bar = xr.open_mfdataset(
    files,
    compat='override',
    preprocess=preprocess
)

# When there is 0 CAPE, MUEL is nan.
# To fix this and make sure the model is trained off all environments,
# set MUEL to 0 where there are nan values.
# Other variables (e.g. CIN) are not so easily set to 0
bar['MUEL'] = xr.where(bar['MUEL'].isnull(), 0, bar['MUEL'])

########################## Thunderstorm ###########################
############################# Indices #############################

# Calculate dew points for thunderstorm parameters
for pressure in ['850', '700', '500']:
    bar[f'dp{pressure}'] = (
        dewpoint_from_specific_humidity(
            pressure=int(pressure) * units.hPa,
            specific_humidity=bar[f'hus{pressure}'] * units('g/g'),
        )
        .metpy.convert_units('K')   # or 'K' depending on your preference
        .metpy.dequantify()            # removes units → returns plain DataArray
    )

# Convective Parameters from RAW TS Climatology Paper
bar['KI'] = bar['ta850'] - bar['ta500'] + bar['dp850'] - (bar['ta700'] - bar['dp700'])
bar['TCD'] = bar['MUEL'] - bar['MULCL']
bar['CCD'] = np.maximum(
    np.minimum(
        bar['MUEL'] - bar['FZL'],
        bar['TCD']
    ),
    np.zeros(bar['FZL'].shape)
)
bar['ATP'] = ((bar['CCD'] - 1000) / 1000) * ((bar['RH24mean'] - 50) / 10)



####################################################################
# ALIGN GRIDS
####################################################################

# Interp BARRA to Heliosat grid, using method="nearest"
# to keep values the same
bar_interp = bar.interp(
    lat=ghi.latitude,
    lon=ghi.longitude,
    method='nearest'
)


# align all datasets so they can be merged
ghi_aligned, rad_aligned, bar_aligned = xr.align(
    ghi,
    ds_rad_interp,
    bar_interp,
    join="inner"
)

####################################################################
# PREPARE DATAFRAME
####################################################################

# merge datasets
ds = xr.merge([ghi_aligned, rad_aligned, bar_aligned])

ds = ds.drop_vars(
    [
        "level_height",
        "model_level_number",
        "sigma",
        "height",
        "crs",
        "pressure",
        "lat",
        "lon",
    ], errors="ignore"
)

df = ds.to_dataframe()

####################################################################
# CALCULATE CSI
####################################################################

# only use data for when solar elevation is above 10 degrees
df = df[df['solar_elevation'] > 10]

# Calculate CSI using pvlib
df['csi'] = clear_sky.csi(
    ghi=df['surface_global_irradiance'],
    time=df.index.get_level_values('time'),
    lat=df.index.get_level_values('latitude'),
    lon=df.index.get_level_values('longitude')
    
)


data = df.dropna()

# Extra satellite predictors (defining here to clean up other scripts).
# Based off NWCSAF cloud retrieval algorithm requirements
data['channel_0013_0015_difference'] = data['channel_0013_brightness_temperature'] - data['channel_0015_brightness_temperature']
data['channel_0011_0013_difference'] = data['channel_0011_brightness_temperature'] - data['channel_0013_brightness_temperature']
data['channel_0007_0013_difference'] = data['channel_0007_brightness_temperature'] - data['channel_0013_brightness_temperature']

start_str = f"{start_date}"[0:10]
end_str = f"{end_date}"[0:10]

data.to_parquet(f'/scratch/er8/cd3022/xgb_datasets/all_training_{start_str}.parquet')
print("DONE!")
