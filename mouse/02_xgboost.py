import xgboost as xgb

from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
import yaml

sys.path.append('/home/548/cd3022/repos/solar-nowcast/modules')
import data_transform
from xgb_preprocessing import prepare_data

from sklearn.metrics import mean_squared_error

model_name = sys.argv[1]

# READ PARQUET TABULAR DATA
data_path = Path('/scratch/er8/cd3022/xgb_datasets/')
df = pd.concat(
    pd.read_parquet(f)
    for f in data_path.glob('*all_training*')
)

# Get configurations from yaml file
with open("/home/548/cd3022/repos/solar-nowcast/configs/mouse/basic.yaml") as f:
    config = yaml.safe_load(f)

model_name = config["model"]["name"]
forecast_lead = config["model"]["forecast_lead"]
test_months = config["model"]["test_months"]
X_vars = config["data"]["predictors"]
target = config["data"]["target"]
y_var = f'{target}_t{forecast_lead}'

# Split into x, y, train, test data
X_train, X_test, y_train, y_test = prepare_data(
    df=df,
    X=X_vars,
    y=y_var,
    test_months=test_months
)


# APPLY LOG TRANSFORM TO CLOUD OPTICAL DEPTH
y_train_log = data_transform.log_transform(y_train)
y_test_log  = data_transform.log_transform(y_test)

# DEFINE MODEL
model = xgb.XGBRegressor(
    random_state=42,
    n_estimators=2000,
    early_stopping_rounds=50,
    learning_rate=0.03,
    eval_metric='rmse',
)

# TRAINING
model.fit(
    X_train, y_train_log,
    eval_set=[(X_train, y_train_log), (X_test, y_test_log)],
    verbose=False
)

# Predict on test set
y_pred = model.predict(X_test)

# Quick evaluation of model performance using correlation and RMSE
correlation = np.corrcoef(y_test_log, y_pred)[0, 1]
rmse = np.sqrt(mean_squared_error(y_test_log, y_pred))

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