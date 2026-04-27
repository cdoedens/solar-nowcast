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
sys.path.append('/home/548/cd3022/repos/solar-nowcast/modules')
import data_transform

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

model_name = sys.argv[1]

# READ PARQUET TABULAR DATA
data_path = Path('/scratch/er8/cd3022/xgb_datasets/')
df = pd.concat(
    pd.read_parquet(f)
    for f in data_path.glob('*all_training*')
)

# ADD MONTH COLUMN TO USE FOR TRAIN/TEST SPLIT
df['month'] = df.index.get_level_values('time').month

# Based off NWCSAF cloud retrieval algorithm requirements
df['channel_0013_0015_difference'] = df['channel_0013_brightness_temperature'] - df['channel_0015_brightness_temperature']
df['channel_0011_0013_difference'] = df['channel_0011_brightness_temperature'] - df['channel_0013_brightness_temperature']
df['channel_0007_0013_difference'] = df['channel_0007_brightness_temperature'] - df['channel_0013_brightness_temperature']


# SPLIT INTO TRAIN AND TEST BASED OFF MONTH
test_months = [2, 6, 10]
train_df = df[~df['month'].isin(test_months)]
test_df  = df[df['month'].isin(test_months)]

# DEFINE PREDICTOR AND TARGET VARIABLES
X_train = train_df.copy()
X_test = test_df.copy()
for n in range(1, 7):
    X_train = X_train.drop(columns=[f'cloud_optical_depth_t{n}'])
    X_test = X_test.drop(columns=[f'cloud_optical_depth_t{n}'])

X_train = X_train.drop(columns=['month'])
X_test = X_test.drop(columns=['month'])
y_train = train_df[['cloud_optical_depth_t1']]
y_test = test_df[['cloud_optical_depth_t1']]


# APPLY LOG TRANSFORM TO CLOUD OPTICAL DEPTH
y_train_log = data_transform.log_transform(y_train)
y_test_log  = data_transform.log_transform(y_test)

# SAVE DATASETS TO BE USED LATER
X_train.to_parquet(data_path / f'train_predictors_{model_name}.parquet')
y_train_log.to_parquet(data_path / f'train_target_{model_name}.parquet')

X_test.to_parquet(data_path / f'test_predictors_{model_name}.parquet')
y_test_log.to_parquet(data_path / f'test_target_{model_name}.parquet')

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