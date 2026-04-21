from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd

import xesmf as xe

import os
import sys

sys.path.append('/home/548/cd3022/repos/solar-nowcast/modules')
import sat_preprocess

import pyearthtools.data as petdata
import pyearthtools.pipeline as petpipe
from pyearthtools.data.time import Petdt
from pyearthtools.pipeline.operations.xarray.join import GeospatialTimeSeriesMerge
import site_archive_nci

date = sys.argv[1]
n = sys.argv[2]

start_year, start_month, start_day = [int(i) for i in date.split('-')]


start_date = datetime(start_year, start_month, start_day)
end_date   = datetime(start_year, start_month, start_day + int(n))

# TO DO: change these to input args
lat_min=-35
lat_max=-32
lon_min=148
lon_max=152


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


rad_list = []

ch_list = [
    'B03',
    'B04',
    # 'B06',
    'B08',
    # 'B11',
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


files = []

year  = start_date.year
month = start_date.month

# Variables from BARRA-C2
variables_of_interest = [
    'huss',
    'hus850',
    'hus700',
    'hus500',
    'psl',
    'tas',
    'ta850',
    'ta700',
    'ta500',
]
for var in variables_of_interest:
    file_path = Path(f'/g/data/ob53/BARRA2/output/reanalysis/AUST-04/BOM/ERA5/historical/hres/BARRA-C2/v1/1hr/{var}/latest/')
    var_file = [f for f in file_path.glob(f'*{year}{month:02d}.nc')][0]
    files.append(var_file)

# Keep just time and region 
def preprocess(ds):
    return ds.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max),
        time=slice(np.datetime64(start_date), np.datetime64(end_date))
    )

# Open BARRA-C2 data
bar = xr.open_mfdataset(
    files,
    compat='override',
    preprocess=preprocess
)

# Interp BARRA-C2 to Heliosat grid, using method="nearest"
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

for n in range(1, 7):
    ds[f'cloud_optical_depth_t{n}'] = (
        ds['cloud_optical_depth'].shift(time=-n)
    )

df = ds.to_dataframe()
data = df.dropna()

start_str = f"{start_date}"[0:10]
end_str = f"{end_date}"[0:10]

data.to_parquet(f'/scratch/er8/cd3022/xgb_datasets/all_training_{start_str}_{end_str}.parquet')
