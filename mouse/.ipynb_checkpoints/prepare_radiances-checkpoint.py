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
end_date   = datetime(year, month, day + int(n))

# TO DO: change these to input args
lat_min=-35
lat_max=-32
lon_min=148
lon_max=152


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



def preprocess(ds):
    return ds.sel(
        latitude=slice(lat_min, lat_max),
        longitude=slice(lon_min, lon_max)
    )[['surface_global_irradiance', 'solar_elevation']]

ghi = xr.open_mfdataset(files, preprocess=preprocess)



rad_list = []

ch_list = [
    'B03',
    'B04',
    # 'B06',
    'B08',
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



ref_ds = rad_list[-1]

interp_list = []

for ds in rad_list:
    if ds.sizes != ref_ds.sizes:
        ds = ds.interp(y=ref_ds.y, x=ref_ds.x)

    # Drop conflicting coords (except for reference)
    if ds is not ref_ds:
        ds = ds.drop_vars(["latitude", "longitude"], errors="ignore")

    interp_list.append(ds)

ds_rad = xr.merge(interp_list)



regridder = xe.Regridder(
    ds_rad,
    ghi,
    method="bilinear",
    reuse_weights=False
)

ds_rad_interp = regridder(ds_rad)

ghi_aligned, ds_rad_interp = xr.align(
    ghi,
    ds_rad_interp,
    join="inner"
)

ds = xr.merge([ghi_aligned, ds_rad_interp])

for n in range(1, 11):
    ds[f'surface_global_irradiance_t{n}'] = (
        ds['surface_global_irradiance'].shift(time=-n)
    )

df = ds.to_dataframe()
data = df.dropna()
data.to_parquet(f'/scratch/er8/cd3022/xgb_datasets/syd_radiances_{start_year}-{start_month:02d}-{start_day:02d}+{n}.parquet')