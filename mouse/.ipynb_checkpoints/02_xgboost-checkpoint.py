import xgboost as xgb

from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt

import os
import sys
import yaml

sys.path.append('/home/548/cd3022/repos/solar-nowcast/modules')
import data_transform
from xgb_preprocess import prepare_data

from sklearn.metrics import mean_squared_error


###############################################################
# LOAD MODEL CONFIGURATION
###############################################################
config_name = sys.argv[1]

# Get configurations from yaml file
with open("/home/548/cd3022/repos/solar-nowcast/configs/mouse/basic.yaml") as f:
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


###############################################################
# LOAD TRAINING AND TESTING DATA
###############################################################

data_path = Path('/scratch/er8/cd3022/xgb_datasets/')
df_train = dd.read_parquet(data_path / "train_months.parquet", columns=X_vars + [y_var])
df_test = dd.read_parquet(data_path / "test_months.parquet", columns=X_vars + [y_var])

# Split into x, y, train, test data
X_train, y_train = prepare_data(
    df=df_train,
    X=X_vars,
    y=y_var,
)

X_test, y_test = prepare_data(
    df=df_test,
    X=X_vars,
    y=y_var,
)

# DEFINE MODEL
model = xgb.XGBRegressor(
    random_state=random_state,
    n_estimators=n_estimators,
    early_stopping_rounds=early_stopping_rounds,
    learning_rate=learning_rate,
    eval_metric=eval_metric,
)

# TRAINING
model.fit(
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

# Predict on test set
y_pred = model.predict(X_test)

# Quick evaluation of model performance using correlation and RMSE
correlation = np.corrcoef(y_test, y_pred)[0, 1]
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Training vs Validation Loss figure
results = model.evals_result()

train_loss = results['validation_0']['rmse']
val_loss = results['validation_1']['rmse']

plt.figure()
plt.plot(train_loss, label='Train RMSE')
plt.plot(val_loss, label='Validation RMSE')

plt.xlabel('Boosting Iterations')
plt.ylabel('RMSE')
plt.title('Training vs Validation Loss')
plt.legend()
plt.savefig(f'/home/548/cd3022/figures/nowcasting/train_val_loss/xgb_{model_name}.png')


model.save_model(f"/scratch/er8/cd3022/xgb_models/{model_name}.json")