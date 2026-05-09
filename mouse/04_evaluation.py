import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import dask.dataframe as dd
import dask.array as da

from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import shap

import yaml
import sys
sys.path.append('/home/548/cd3022/repos/solar-nowcast/modules')
import data_transform
from xgb_preprocess import prepare_data
from gso_preprocess import datetime_to_lead_time

model = xgb.XGBRegressor()

config_name = sys.argv[1]

############################################################
# Load model and data
############################################################

# Get configurations from yaml file
with open(f"/home/548/cd3022/repos/solar-nowcast/configs/mouse/{config_name}.yaml") as f:
    config = yaml.safe_load(f)
# Model
model_name = config["model"]["name"]
forecast_lead = config["model"]["forecast_lead"]
# Parameters
random_state = config["model"]["parameters"]["random_state"]
n_estimators = config["model"]["parameters"]["n_estimators"]
early_stopping_rounds = config["model"]["parameters"]["early_stopping_rounds"]
learning_rate = config["model"]["parameters"]["learning_rate"]
eval_metric = config["model"]["parameters"]["eval_metric"]

# Variables 
X_vars = config["data"]["predictors"]
target = config["data"]["target"]
y_var = f'{target}_t{forecast_lead}'
all_vars = X_vars + [y_var]

# Model json
model.load_model(f"/scratch/er8/cd3022/xgb_models/{model_name}.json")

# Load data from parquet files
data_path = Path('/scratch/er8/cd3022/xgb_datasets/')

test_files = []

for f in data_path.glob("all_training_month*"):
    month = f.name.split("_")[-1][:2]  # adjust to your naming

    if month in ['02', '06', '10']:
        test_files.append(str(f))

df  = pd.read_parquet(test_files, columns=all_vars)

# Get model inputs
X = df[X_vars]

# Add model predictions to dataframe
df[f'forecast_t{forecast_lead}'] = model.predict(X)

# convert to xarray dataset
ds_xgb = df.to_xarray()

############################################################
# Load GSO Optical Flow forecast
############################################################

gso_files = []
for month in np.unique(ds_xgb.time.dt.month):
    for day in np.unique(ds_xgb.time.dt.day):
        gso_path = Path(f'/scratch/er8/mm4602/gso/output/nwc_20240701_20251231/2025/{month:02d}/{day:02d}')
        gso_files.extend([f for f in gso_path.glob('*.nc')])

ds_gso = xr.open_mfdataset(
    gso_files,
    preprocess=datetime_to_lead_time,
    combine='nested',
    concat_dim='time',
    parallel=True,
    chunks='auto'
)

# Align datasets to ensure like-for-like comparison
ds_gso, ds_xgb = xr.align(
    ds_gso,
    ds_xgb,
    join="inner"
)

############################################################
# Performance Analysis
############################################################

def rmse(obs, fcst):
    mse = np.mean((fcst - obs) ** 2)
    rmse = np.sqrt(mse)
    return rmse

# TO DO:
# add functions for other metrics, e.g.
# SSIM
# FFS

# RMSE for all times

obs = ds_xgb.cloud_optical_depth
fcst_xgb = ds_xgb.forecast
fcst_gso = ds_gso.sel(forecast_time='01:00:00').cloud_optical_depth

rmse_xgb = rmse(obs, fcst_xgb)
print(f"XGBoost RMSE: {rmse_xgb.item()}")

rmse_gso = rmse(obs, fcst_gso)
rmse_gso = rmse_gso.compute()
print(f"Optical Flow RMSE: {rmse_gso.item()}")

# RMSE by month

# RMSE by month

for month, month_ds in ds_xgb.groupby('time.month'):
    
    obs = month_ds[f'csi_t{forecast_lead}']
    fcst_xgb = month_ds[f'forecast_t{forecast_lead}']
    fcst_gso = ds_xgb['csi_gso'].sel(time=month_ds.time)
    
    rmse_xgb = rmse(obs, fcst_xgb)
    print(f"Month: {month:02d}")
    print(f"XGBoost RMSE: {rmse_xgb.item()}")
    
    rmse_gso = rmse(obs, fcst_gso)
    print(f"Optical Flow RMSE: {rmse_gso.item()}")


# By CSI magnitude

csi_ranges = [
    (0.25, 0.5),
    (0.5, 0.75),
    (0.75, 10),
]

for csi_min, csi_max in csi_ranges:
    ds_range = ds_xgb.where((ds_xgb[f'csi_t{forecast_lead}'] > csi_min) & (ds_xgb[f'csi_t{forecast_lead}'] < csi_max))
    obs = ds_range[f'csi_t{forecast_lead}']
    fcst_xgb = ds_range[f'forecast_t{forecast_lead}']
    fcst_gso = ds_range['csi_gso']

    rmse_xgb = rmse(obs, fcst_xgb)
    print(f"CSI Range: {csi_min} - {csi_max}")
    print(f"XGBoost RMSE: {rmse_xgb.item()}")
    
    rmse_gso = rmse(obs, fcst_gso)
    print(f"Optical Flow RMSE: {rmse_gso.item()}")


# By change in CSI

ds_xgb['csi_delta'] = ds_xgb[f'csi_t{forecast_lead}'] - ds_xgb['csi']

delta_range = [
    (-10, -0.1),
    (-0.1, 0.1),
    (0.1, 10)
]

for delta_min, delta_max in delta_range:
    ds_range = ds_xgb.where((ds_xgb['csi_delta'] > delta_min) & (ds_xgb['csi_delta'] < delta_max))

    obs = ds_range[f'csi_t{forecast_lead}']
    fcst_xgb = ds_range[f'forecast_t{forecast_lead}']
    fcst_gso = ds_range['csi_gso']

    rmse_xgb = rmse(obs, fcst_xgb)
    print(f"CSI Delta Range: {csi_min} - {csi_max}")
    print(f"XGBoost RMSE: {rmse_xgb.item()}")
    
    rmse_gso = rmse(obs, fcst_gso)
    print(f"Optical Flow RMSE: {rmse_gso.item()}")

############################################################
# SHAP Results
############################################################

explainer = shap.Explainer(model)

idx = np.random.choice(len(X), size=20000, replace=False)
X_sample = X.iloc[idx]

shap_values = explainer(X_sample)

shap.plots.bar(shap_values)

shap.plots.beeswarm(shap_values)