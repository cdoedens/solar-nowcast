import xarray as xr
from pathlib import Path
import re
from datetime import datetime, timedelta
import pandas as pd


def get_ancil(file_path):
    """
    Retrieve the ancillary file containing coordinate mappings for satellite data.

    Parameters:
        base_dir (str or Path): file path to the satellite dataset

    Returns:
        ancillary xarray.dataset
    """
    file_name = file_path.name
    match = re.search(r'GEOS141_(\d+)-HIMAWARI', file_name)
    assert match is not None, f"Filename format invalid: {file_name}"
    
    resolution = match.group(1)
    
    ancil_file = Path(f'/g/data/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest/ancillary/00000000000000-P1S-ABOM_GEOM_SENSOR-PRJ_GEOS141_{resolution}-HIMAWARI8-AHI.nc')

    return xr.open_dataset(ancil_file).isel(time=0)







def get_files(channel, start_time, end_time, step_minutes=10):
    """
    Retrieve all files between start_time and end_time based on directory structure:
    base_dir/YYYY/MM/DD/HHMM/

    Parameters:
        base_dir (str or Path): Root directory
        start_time (datetime)
        end_time (datetime)
        step_minutes (int): Time step between folders (default 10)

    Returns:
        list of Path objects
    """
    # correct for UTC time diff
    start_time -= timedelta(hours=10)
    end_time -= timedelta(hours=10)
    
    base_dir = '/g/data/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest'
    base_dir = Path(base_dir)
    current = start_time
    files = []

    while current <= end_time:
        dir_path = base_dir / current.strftime("%Y/%m/%d/%H%M")

        if dir_path.exists():
            files.extend([f for f in dir_path.glob(f'*OBS_{channel}*.nc')])

        current += timedelta(minutes=step_minutes)

    return files





def get_sat_reg(ds, ancil, lat_min, lat_max, lon_min, lon_max, coords=False):
    """
    Use the ancillary file to select just a lat/lon region bounds.

    Parameters:
        ds: the satellite dataset of interest
        ancil: the ancillary dataset containing coordinate mapping
        lat_min, lat_max, lon_min, lon_max: boundaries for the region of interest

    Returns:
        ds cropped to the coordinate bounds
    """
    reg = ancil.where(
        (ancil.lat >= lat_min) & (ancil.lat <= lat_max) & 
        (ancil.lon >= lon_min) & (ancil.lon <= lon_max),
        drop=True
    )

    ds_reg = ds.sel(y=reg.y.values, x=reg.x.values, method='nearest')

    if not coords:
        return ds_reg
    
    return ds_reg.assign_coords({
        "latitude": (("y", "x"), reg.lat.values),
        "longitude": (("y", "x"), reg.lon.values),
    })



    

def read_himawari_channel(channel, start, end, lat_min, lat_max, lon_min, lon_max, coords=False):
    '''
    Retrieve Himawari-8/9 radiance data from an individual channel for a certain time period and region.

    INPUTS
    channel (str): name of the AHI channel
    start, end (datetime): time bounds to retrieve data for
    lat_min, lat_max, lon_min, lon_max: region boundaries
    '''
    
    files = get_files(channel, start, end)

    # get ancil once, same for all files
    ancil = get_ancil(files[0])

    def preprocess(ds):
        return get_sat_reg(ds, ancil, lat_min, lat_max, lon_min, lon_max, coords)

    return xr.open_mfdataset(
        files,
        preprocess=preprocess,
        concat_dim='time',
        combine='nested',
        data_vars='minimal',
        coords='minimal',
        compat='override'
    )
    



def read_satproduct(files, lat_min, lat_max, lon_min, lon_max, coords=False):

    ancil_file = Path(f'/g/data/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest/ancillary/00000000000000-P1S-ABOM_GEOM_SENSOR-PRJ_GEOS141_2000-HIMAWARI8-AHI.nc')
    ancil = xr.open_dataset(ancil_file).isel(time=0)

    def preprocess(ds):
        # add missing time dimension using time stored as attr
        ds = ds.expand_dims(
            time=[pd.Timestamp(ds.attrs['nominal_product_time'])]
        )

        # rename dim for consistency with earlier functions
        ds = ds.rename({
            'ny':'y',
            'nx':'x'
        })
        return get_sat_reg(ds, ancil, lat_min, lat_max, lon_min, lon_max, coords)

    return xr.open_mfdataset(
        files,
        preprocess=preprocess,
        concat_dim='time',
        combine='nested',
        data_vars='minimal',
        coords='minimal',
        compat='override'
    )





    