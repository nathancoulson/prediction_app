import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime

from random import randint
from numpy import array
from numpy import argmax
from random import randint

from math import sin
from math import pi
from math import exp
from random import randint
from random import uniform

from numpy import array
from math import ceil
from math import log10

from random import random
from numpy import cumsum
from numpy import array_equal

import json
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler

import re

path = "../Datasets/"

log_df = pd.read_parquet(path + "logs.parquet")

log_df.dropna(inplace=True)

# Feature transformations

# Convert resp_time and bytes_setn into float

log_df.resp_time = log_df.resp_time.astype("float")
log_df.bytes_sent = log_df.bytes_sent.astype("float")

app_cols = [col for col in log_df.columns if "app_" in col]

# Scale key cols between 0 and 1

log_df["resp_time"] = MinMaxScaler().fit_transform(
    log_df[["resp_time"]]
)
log_df[app_cols] = MinMaxScaler().fit_transform(
    log_df[app_cols]
)


log_df.drop(
    columns=["bytes_sent", "resp_code", "url", "datetime"],
    inplace=True,
)

# Extract and prepare data

y = log_df.resp_time

X = log_df.drop(columns=["resp_time"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_train = np.ascontiguousarray(X_train)
y_train = np.ascontiguousarray(y_train)

X_test = np.ascontiguousarray(X_test)
y_test = np.ascontiguousarray(y_test)

# Prep grid search

models_final = {
    "Ridge": {
        "model": Ridge(),
        "params": {
            "alpha": [1, 0.1, 0.01, 0.001, 0],
            "fit_intercept": [True, False],
            "solver": [
                "svd",
                "cholesky",
                "sparse_cg",
                "sag",
            ],
        },
    },
    "LinearRegression": {
        "model": LinearRegression(),
        "params": {"fit_intercept": [True, False]},
    },
    "Lasso": {
        "model": Lasso(),
        "params": {
            "alpha": [1, 0.1, 0.01, 0.001, 0.0001, 0],
            "fit_intercept": [True, False],
        },
    },
    "ElasticNet": {
        "model": ElasticNet(),
        "params": {
            "max_iter": [1, 5, 10],
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],
            "l1_ratio": np.arange(0.0, 1.0, 0.2),
        },
    },
    "RandomForestRegressor": {
        "model": RandomForestRegressor(),
        "params": {
            "bootstrap": [True, False],
            "max_depth": [2, 3, 5, 10, 20, 40, 80, None],
            "max_features": ["sqrt"],
            "min_samples_leaf": [1, 2, 4],
            "min_samples_split": [2, 5, 10],
            "n_jobs": [-1],
            "n_estimators": [10, 50, 100, 400, 1000, 2000],
        },
    },
    "KNeighborsRegressor": {
        "model": KNeighborsRegressor(),
        "params": {
            "n_neighbors": [2, 4, 8, 16, 32],
            "weights": ["uniform", "distance"],
            "n_jobs": [-1],
            "algorithm": [
                "auto",
                "ball_tree",
                "kd_tree",
                "brute",
            ],
        },
    },
}


models_dict = dict()

for model, params in models_final.items():

    print("Running gridsearch on " + str(model))

    search = GridSearchCV(
        params["model"], params["params"], iid=False, cv=5
    )

    search.fit(X_train, y_train)

    y_pred = search.predict(X_test)

    models_dict[params["model"]] = {
        "Best score": search.best_score_,
        "Best params": search.best_params_,
        "Best estimator": search.best_estimator_,
        "Mean Absolute Error": mean_absolute_error(
            y_test, y_pred
        ),
        "Mean Squared Error": mean_squared_error(
            y_test, y_pred
        ),
        "Root Mean Squared Error": np.sqrt(
            mean_squared_error(y_test, y_pred)
        ),
        "CV results": search.cv_results_,
        "Datetime": "_".join(
            str(datetime.datetime.now()).split()
        ),
    }

    print(models_dict)

    with open(
        "../Models/super_static_model_results.pkl", "wb"
    ) as f:
        pickle.dump(models_dict, f)

    print(str(params["model"]) + " model done!")
