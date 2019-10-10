import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
import re


def create_request_model_table(log_df, path):
    """
    Function to transform log dataframe into request tables for modelling
    """

    # Feature transformations

    # Create categorical variables for url requests

    log_df = pd.get_dummies(log_df, columns=["url"])
    url_features = [
        col for col in log_df.columns if "url" in col
    ]
    req_df = log_df[url_features]

    # Create sequence prediction outcome variable -> next 250 requests

    for url in url_features:
        req_df["CS_" + url] = req_df[url].rolling(250).sum()

    cumsum_cols = [
        col for col in req_df.columns if "CS_" in col
    ]

    # Delete NaN rows

    req_df.dropna(inplace=True)

    # Shift output variable 250 places down

    req_df[cumsum_cols] = req_df[cumsum_cols].shift(-250)

    # Delete NaN rows again

    req_df.dropna(inplace=True)

    # Scale CS cols between 0 and 1

    req_df[cumsum_cols] = MinMaxScaler().fit_transform(
        req_df[cumsum_cols]
    )

    # Split 50/50 into train and test sets (keeping distribution)

    train_df = req_df.iloc[::2]  # even

    test_df = req_df.iloc[1::2]  # odd

    # Create subset (every 2th request) and trim

    # trim to size

    train_df = train_df.iloc[0:2241000]
    test_df = test_df.iloc[0:2241000]

    train_df.to_parquet(path + "train.parquet")
    test_df.to_parquet(path + "test.parquet")

    with open(path + "cumsum_cols.pkl", "wb") as f:
        pickle.dump(cumsum_cols, f)

    print(
        "Modelling tables created and saved at path: "
        + path
    )
