import pandas as pd
import os
import sys
import numpy as np
import datetime
from numpy import array
import re
from sklearn.preprocessing import MinMaxScaler

def create_resp_time_model_table(log_df, path):
    '''
    Function to generate response time modelling table for supervised model - rolling data used
    '''
    
    # IMPORTANT!! - deleting all logs without a response time (errors) - this loses key information but should simplify the problem while preserving the basic relationship

    log_df.dropna(inplace=True)

    # Feature transformations

    log_df.resp_time = log_df.resp_time.astype("float")
    log_df.bytes_sent = log_df.bytes_sent.astype("float")

    log_df.datetime = pd.to_datetime(log_df.datetime,
                                        format="%Y-%m-%d %H:%M:%S")

    log_df = pd.get_dummies(log_df, columns = ['resp_code'])
    log_df = pd.get_dummies(log_df, columns = ['url'])

    # Create column lists

    url_features = [col for col in log_df.columns if "url" in col]
    app_cols = [col for col in log_df.columns if "app_" in col]
    resp_codes = [col for col in log_df.columns if "resp_code" in col]

    # Create sequence prediction outcome variable -> next 250 requests

    for url in url_features:
         log_df["CS_" + url] = log_df[url].rolling(250).sum()

    cumsum_cols = [col for col in log_df.columns if "CS_" in col]

    # Create rolling average request response time (over the next 250 requests)

    log_df["av_rolling_resp_time_250"] = log_df["resp_time"].rolling(250).sum() / 250
    log_df.dropna(inplace=True)

    # Shift request_mix and response_time variables 250 places down

    log_df[cumsum_cols] = log_df[cumsum_cols].shift(-250)
    log_df["av_rolling_resp_time_250"] = log_df["av_rolling_resp_time_250"].shift(-250)
    log_df.dropna(inplace=True)

    # Scale key cols between 0 and 1

    log_df[cumsum_cols] = MinMaxScaler().fit_transform(log_df[cumsum_cols])
    log_df["av_rolling_resp_time_250"] = MinMaxScaler().fit_transform(log_df[["av_rolling_resp_time_250"]])
    log_df[app_cols] = MinMaxScaler().fit_transform(log_df[app_cols])
    
    # Drop unnecessary columns
    
    columns_to_drop = url_features + resp_codes + ["resp_time", "bytes_sent", "datetime"]
    log_df.drop(columns=columns_to_drop, inplace=True)

    # Save to parquet
    
    log_df.to_parquet(path + 'resp_df.parquet')
    
    print("Response time modelling table saved at path: " + path)
