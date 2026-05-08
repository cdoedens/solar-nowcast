from pathlib import Path
import matplotlib.pyplot as plt

from dask.distributed import Client
import dask.dataframe as dd
import xgboost.dask as dxgb

import os
import sys
import yaml

if __name__ == '__main__':
    # Start dask client for distributed training
    client = Client(
        n_workers=8,
        threads_per_worker=1
    )

    # match partitions with workers
    num_partitions = 64

    ###############################################################
    # LOAD MODEL CONFIGURATION
    ###############################################################
    config_name = sys.argv[1]
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
    
    
    ###############################################################
    # LOAD TRAINING AND TESTING DATA
    ###############################################################
    data_path = Path('/scratch/er8/cd3022/xgb_datasets/')
    
    # Collect monthly files for training and testing separately
    train_files = []
    test_files = []
    for f in data_path.glob("all_training_month*"):
        month = f.name.split("_")[-1][:2]  # adjust to your naming
    
        if month in ['02', '06', '10']:
            test_files.append(str(f))
        else:
            train_files.append(str(f))
    
    df_train = (
        dd.read_parquet(train_files, columns=all_vars)
        .astype("float32")
        .repartition(npartitions=num_partitions)
        .persist()
    )

    df_test = (
        dd.read_parquet(test_files, columns=all_vars)
        .astype("float32")
        .repartition(npartitions=num_partitions)
        .persist()
    )

    # Split AFTER repartitioning
    X_train = df_train[X_vars]
    y_train = df_train[y_var]

    X_test = df_test[X_vars]
    y_test = df_test[y_var]

    # monitor set
    train_monitor = (
        df_train.sample(frac=0.001)
        .persist()
    )

    X_train_monitor = train_monitor[X_vars]
    y_train_monitor = train_monitor[y_var]
    ###############################################################
    # XGBoost
    ###############################################################
    # Define model
    model = dxgb.DaskXGBRegressor(
        random_state=random_state,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric=eval_metric,
        tree_method="hist",
    )
    
    # TRAIN
    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_train_monitor, y_train_monitor),
            (X_test, y_test),
        ],
        verbose=False,
    )
    
    ###############################################################
    # Evaluate traiing
    ###############################################################
    
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
    
    ###############################################################
    # Save model
    ###############################################################
    
    model.save_model(f"/scratch/er8/cd3022/xgb_models/{model_name}.json")
    print("DONE!")
