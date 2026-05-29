from pathlib import Path
import os
import sys
import yaml
import json

import optuna
import numpy as np

from dask.distributed import Client
import dask.dataframe as dd

import xgboost.dask as dxgb
from sklearn.metrics import mean_squared_error






###############################################################################
# RMSE
###############################################################################
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":

    ###########################################################################
    # DASK
    ###########################################################################
    client = Client(
        n_workers=1,
        threads_per_worker=12,
    )

    num_partitions = 2

    ###########################################################################
    # LOAD CONFIG
    ###########################################################################
    config_name = sys.argv[1]

    with open(
        f"/home/548/cd3022/repos/solar-nowcast/configs/mouse/{config_name}.yaml"
    ) as f:
        config = yaml.safe_load(f)

    # MODEL
    model_name = config["model"]["name"]
    forecast_lead = config["model"]["forecast_lead"]

    # DATA
    X_vars = config["data"]["predictors"]
    target = config["data"]["target"]

    y_var = f"{target}_t{forecast_lead}"
    all_vars = X_vars + [y_var]

    ###########################################################################
    # LOAD DATA
    ###########################################################################
    data_path = Path("/scratch/er8/cd3022/xgb_datasets/")

    train_files = []
    test_files = []
    validation_files = []

    for f in data_path.glob("all_training_month*"):

        month = f.name.split("_")[-1][:2]

        if month in ["02", "06", "10"]:
            test_files.append(str(f))
        elif month in ["12", "04", "08"]:
            validation_files.append(str(f))
        else:
            train_files.append(str(f))

    df_train = (
        dd.read_parquet(train_files, columns=all_vars)
        .sample(frac=0.1, random_state=42)
        .astype("float32")
        .repartition(npartitions=num_partitions)
        .persist()
    )

    df_validation = (
        dd.read_parquet(validation_files, columns=all_vars)
        .sample(frac=0.1, random_state=42)
        .astype("float32")
        .repartition(npartitions=num_partitions)
        .persist()
    )

    df_test = (
        dd.read_parquet(test_files, columns=all_vars)
        .sample(frac=0.1, random_state=42)
        .astype("float32")
        .repartition(npartitions=num_partitions)
        .persist()
    )

    X_train = df_train[X_vars]
    y_train = df_train[y_var]

    X_validation = df_validation[X_vars]
    y_validation = df_validation[y_var]

    X_test = df_test[X_vars]
    y_test = df_test[y_var]

    # monitor set
    train_monitor = (
        df_train.sample(frac=0.01)
        .persist()
    )

    X_train_monitor = train_monitor[X_vars]
    y_train_monitor = train_monitor[y_var]


    ###########################################################################
    # OPTUNA OBJECTIVE
    ###########################################################################
    def objective(trial):

        params = {
            "random_state": 42,
            "tree_method": "hist",

            # Core parameters
            "max_depth": trial.suggest_int("max_depth", 4, 12),

            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.2,
                log=True,
            ),

            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                1,
                10,
            ),

            "subsample": trial.suggest_float(
                "subsample",
                0.5,
                1.0,
            ),

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.5,
                1.0,
            ),

            # Regularization
            "gamma": trial.suggest_float(
                "gamma",
                0.0,
                1.0,
            ),

            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-4,
                10.0,
                log=True,
            ),

            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1e-3,
                10.0,
                log=True,
            ),

            # Large number + early stopping
            "n_estimators": 500,

            "early_stopping_rounds": 30,

            "eval_metric": "rmse",
            "device": "cuda",

        }

        #######################################################################
        # MODEL
        #######################################################################
        model = dxgb.DaskXGBRegressor(**params)

        #######################################################################
        # TRAIN
        #######################################################################
        
        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_train_monitor, y_train_monitor),
                (X_validation, y_validation),
            ],
            verbose=False,
        )

        #######################################################################
        # PREDICT
        #######################################################################
        y_pred = model.predict(X_validation)

        # convert from dask -> numpy
        y_pred = y_pred.compute()
        y_true = y_validation.compute()

        score = rmse(y_true, y_pred)

        print(f"Trial {trial.number}: RMSE = {score:.5f}")

        return score

    ###########################################################################
    # OPTUNA STUDY
    ###########################################################################
    study = optuna.create_study(
        direction="minimize",
        study_name=f"{model_name}_lead{forecast_lead}",
    )

    study.optimize(
        objective,
        n_trials=100,
        show_progress_bar=True,
    )

    ###########################################################################
    # RESULTS
    ###########################################################################
    print("\nBEST PARAMETERS")
    print(study.best_params)

    print("\nBEST RMSE")
    print(study.best_value)

    ###########################################################################
    # SAVE RESULTS
    ###########################################################################
    output_dir = Path("/scratch/er8/cd3022/xgb_models/optuna")
    output_dir.mkdir(exist_ok=True)

    with open(
        output_dir / f"{model_name}_lead{forecast_lead}_best_params.json",
        "w",
    ) as f:
        json.dump(study.best_params, f, indent=4)

    print("\nSaved best parameters.")