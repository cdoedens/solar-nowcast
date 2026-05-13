import xgboost as xgb
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import dask.dataframe as dd
import cartopy.crs as ccrs

import sys
import yaml

config_name = sys.argv[1]
model = xgb.XGBRegressor()

###############################################################################################
# LOAD MODEL AND DATA
###############################################################################################

# Get configurations from yaml file
with open(f"/home/548/cd3022/repos/solar-nowcast/configs/mouse/{config_name}.yaml") as f:
    config = yaml.safe_load(f)
# Model
model_name = config["model"]["name"]
model.load_model(f"/scratch/er8/cd3022/xgb_models/{model_name}.json")

forecast_lead = config["model"]["forecast_lead"]

# Data vars
X_vars = config["data"]["predictors"]
target = config["data"]["target"]
y_var = f'{target}_t{forecast_lead}'
all_vars = X_vars + [y_var]

# Load data
data_path = Path('/scratch/er8/cd3022/xgb_datasets/')

test_files = []

for f in data_path.glob("all_training_month*"):
    month = f.name.split("_")[-1][:2]  # adjust to your naming

    if month in ['02', '06', '10']:
        test_files.append(str(f))

df  = dd.read_parquet(test_files, columns=all_vars)
df = df.compute()

# Make XGB forecast
X = df[X_vars]
y = df[y_var]

preds = model.predict(X)
fcst = f"xgb_t{forecast_lead}"
df[fcst] = preds

# Convert to xarray for spatial plotting
ds_xgb = df.to_xarray()
ds_xgb = ds_xgb.compute()

# Load GSO Optical Flow (CSI already computed)
ds_gso = xr.open_dataset("/scratch/er8/cd3022/xgb_datasets/gso_csi_testing.nc")

###############################################################################################
# PLOT CASE STUDIES AS SPATIAL CSI MAPS
###############################################################################################

# case studies to plot
times = ['2025-02-05T01:00', '2025-02-08T01:00', '2025-06-07T01:00', '2025-06-03T01:00', '2025-10-06T03:00', '2025-10-10T01:00']

# Plotting
ncols = len(times)
vmin=0
vmax=1
syd_coords = (-33.876, 151.211)
size = 3
figsize=(size*ncols,size)
fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
for i, t in enumerate(times):
    ax[i].pcolormesh(
        ds_xgb.longitude,
        ds_xgb.latitude,
        ds_xgb[fcst].sel(time=t),
        cmap='viridis', shading='auto',
        vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree()
    )
    ax[i].scatter(syd_coords[1], syd_coords[0], color='red', marker='o', s=100)
    ax[i].text(syd_coords[1] - 2, syd_coords[0], 'Sydney', color='black')
    ax[i].coastlines()
    ax[i].set_title(t, fontsize=12)
    ax[i].set_ylabel(model_name, rotation=0, labelpad=40, va='center')
fig.text(
    0.02,          # x position in figure coordinates
    0.5,           # y position
    f"XGB: {model_name}",
    rotation=90,
    va='center',
    ha='right',
    fontsize=14
)
plt.tight_layout(rect=[0.02, 0, 1, 1])
plt.savefig(f"/home/548/cd3022/figures/nowcasting/case_studies/xgb_{model_name}.png")