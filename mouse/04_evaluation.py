import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import xarray as xr
import dask.dataframe as dd

from skimage.metrics import structural_similarity as ssim
import shap

import yaml
import os
import sys

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

# Data
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

df = dd.read_parquet(test_files, columns=all_vars)
df = df.compute()

# Get model inputs
X = df[X_vars]

# Add model predictions to dataframe
df[f'xgb_t{forecast_lead}'] = model.predict(X)

# convert to xarray dataset
ds_xgb = df.to_xarray()
# load into memory
ds_xgb = ds_xgb.compute()

############################################################
# Load GSO Optical Flow forecast
############################################################

# Load GSO Optical Flow forecast
# Data has been processed and prepared for evaluation purposes already
# TO DO:
# Write script with the GSO preprocessing for replicacability
ds_gso = xr.open_dataset("/scratch/er8/cd3022/xgb_datasets/gso_csi_testing.nc")


############################################################
# Performance Analysis
############################################################

# Record performance of model in different conditions
mod_metrics = {}
gso_metrics = {}

# Use shape to get feature importance
# currently not saving results, need to think of method to collect results across many models
explainer = shap.Explainer(model)

def rmse(obs, fcst):
    mse = np.mean((fcst - obs) ** 2)
    rmse = np.sqrt(mse)
    return rmse

# TO DO:
# add functions for other metrics, e.g.
# SSIM
# FFS

def shap_analysis(ds, df, X_vars, explainer):
    '''
    Produce SHAP beeswarm plot using data from ds (matched with already loaded
    df to get tabular data), the X_vars from yaml config, and the shap explainer
    '''

    times = pd.to_datetime(ds.time.values)
    
    shap_df = df[df.index.get_level_values('time').isin(times)]

    # select the predictor variables
    X = shap_df[X_vars]

    # get smaller sample size to reduce shap load time
    n = min(len(X), 20000)
    idx = np.random.choice(len(X), size=n, replace=False)
    X_sample = X.iloc[idx]

    # Use sample to get SHAP explainer values
    shap_values = explainer(X_sample)

    # Plot results
    # shap.plots.bar(shap_values)
    shap.plots.beeswarm(shap_values)
    return

############################################################
# Total Performance
############################################################

obs = ds_xgb[f'csi_t{forecast_lead}']
fcst_xgb = ds_xgb[f'xgb_t{forecast_lead}']
fcst_gso = ds_gso['csi_gso']

rmse_xgb = rmse(obs, fcst_xgb)
rmse_gso = rmse(obs, fcst_gso)

mod_metrics['rmse_all'] = rmse_xgb.item()
gso_metrics['rmse_all'] = rmse_gso.item()

############################################################
# Performance by Month
############################################################

for month, month_ds in ds_xgb.groupby('time.month'):
    
    obs = month_ds[f'csi_t{forecast_lead}']
    fcst_xgb = month_ds[f'xgb_t{forecast_lead}']
    fcst_gso = ds_gso['csi_gso'].sel(time=month_ds.time)
    
    rmse_xgb = rmse(obs, fcst_xgb)
    rmse_gso = rmse(obs, fcst_gso)
    # shap_analysis(
    #     ds=month_ds,
    #     df=df,
    #     X_vars=X_vars,
    #     explainer=explainer
    # )

    mod_metrics[f'rmse_month_{month:02d}'] = rmse_xgb.item()
    gso_metrics[f'rmse_month_{month:02d}'] = rmse_gso.item()


############################################################
# Performance by Daily Mean CSI
############################################################

daily_mean_csi = ds_xgb[f'csi_t{forecast_lead}'].resample(time='1D').mean()
daily_regional_mean = daily_mean_csi.mean(['latitude', 'longitude'])

csi_ranges = [
    (0, 0.6, 'low'),
    (0.6, 0.8, 'med'),
    (0.8, 10, 'high'),
]

# Daily mean CSI
daily_mean_csi = ds_xgb[f'csi_t{forecast_lead}'].resample(time='1D').mean()

for csi_min, csi_max, name in csi_ranges:

    # Select days whose DAILY MEAN CSI falls in range
    valid_days = daily_regional_mean.time.where(
        (daily_regional_mean > csi_min) &
        (daily_regional_mean < csi_max),
        drop=True
    )
    # Get dates only
    valid_dates = valid_days.dt.floor('D')

    # Match all timestamps belonging to those days
    mask = ds_xgb.time.dt.floor('D').isin(valid_dates)
    ds_range = ds_xgb.where(mask, drop=True)

    obs = ds_range[f'csi_t{forecast_lead}']
    fcst_xgb = ds_range[f'xgb_t{forecast_lead}']
    fcst_gso = ds_gso['csi_gso'].sel(time=ds_range.time)


    rmse_xgb = rmse(obs, fcst_xgb)
    print(f"CSI Range: {csi_min} - {csi_max}")
    print(f"XGBoost RMSE: {rmse_xgb.item()}")
    
    rmse_gso = rmse(obs, fcst_gso)
    print(f"Optical Flow RMSE: {rmse_gso.item()}")
    # shap_analysis(
    #     ds=ds_range,
    #     df=df,
    #     X_vars=X_vars,
    #     explainer=explainer
    # )
    mod_metrics[f'rmse_csi_{name}'] = rmse_xgb.item()
    gso_metrics[f'rmse_csi_{name}'] = rmse_gso.item()


############################################################
# Performance by CSI Rate of Change
############################################################

reg_mean = ds_xgb.mean(['latitude', 'longitude'])
reg_mean_delta = reg_mean[f'csi_t{forecast_lead}'] - reg_mean['csi']

delta_range = [
    (-10, -0.1, 'shrink'),
    (-0.1, 0.0, 'stable'),
    (0.0, 10, 'grow')
]

for delta_min, delta_max, name in delta_range:
    
    valid_times = reg_mean_delta.time.where(
        (reg_mean_delta > delta_min) &
        (reg_mean_delta < delta_max),
        drop=True
    )

    # Match all timestamps belonging to those days
    mask = ds_xgb.time.isin(valid_times)
    ds_range = ds_xgb.where(mask, drop=True)

    obs = ds_range[f'csi_t{forecast_lead}']
    fcst_xgb = ds_range[f'xgb_t{forecast_lead}']
    fcst_gso = ds_gso['csi_gso'].sel(time=ds_range.time)


    rmse_xgb = rmse(obs, fcst_xgb)
    print(f"CSI Range: {delta_min} - {delta_max}")
    print(f"XGBoost RMSE: {rmse_xgb.item()}")
    
    rmse_gso = rmse(obs, fcst_gso)
    print(f"Optical Flow RMSE: {rmse_gso.item()}")
    # shap_analysis(
    #     ds=ds_range,
    #     df=df,
    #     X_vars=X_vars,
    #     explainer=explainer
    # )
    mod_metrics[f'rmse_delta_{name}'] = rmse_xgb.item()
    gso_metrics[f'rmse_delta_{name}'] = rmse_gso.item()

############################################################
# Save Results
############################################################

df_metrics = pd.DataFrame(mod_metrics, index=[model_name])
df_metrics_gso = pd.DataFrame(gso_metrics, index=['GSO_OF'])

save_path = Path('/scratch/er8/cd3022/xgb_eval/')
os.makedirs(save_path, exist_ok=True)

df_metrics.to_csv(f'/scratch/er8/cd3022/xgb_eval/{model_name}_metrics.csv')
df_metrics_gso.to_csv('/scratch/er8/cd3022/xgb_eval/gso_of_metrics.csv')