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


# Load GHI data

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
    'solar_elevation'
]

# keep just the region and variables during preprocessing
def preprocess(ds):
    return ds.sel(
        latitude=slice(lat_min, lat_max),
        longitude=slice(lon_min, lon_max)
    )[helio_vars]


# Open the data
ghi = xr.open_mfdataset(files, preprocess=preprocess)

# Convert to dataframe
df = ghi.to_dataframe()

# only use data for when solar elevation is above 10 degrees
df = df[df['solar_elevation'] > 10]

# Calculate CSI using pvlib
df['csi'] = clear_sky.csi(
    ghi=df['surface_global_irradiance'],
    time=df.index.get_level_values('time'),
    lat=df.index.get_level_values('latitude'),
    lon=df.index.get_level_values('longitude')
    
)

# just keep CSI
data = df[['csi']].dropna()


# Save data
start_str = f"{start_date}"[0:10]

data.to_parquet(f'/scratch/er8/cd3022/xgb_datasets/csi_{start_str}.parquet')
