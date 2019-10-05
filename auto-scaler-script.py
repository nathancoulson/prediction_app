import pandas as pd
import os
import re
import numpy as np
import datetime
from random import randint
from numpy import array
import pickle
import sys
import itertools

from sklearn.linear_model import Ridge
import sklearn
import keras
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from ETL_Pipeline_Module.data_extraction import *
from Auto_Scaler_Module.auto_scaler_functions import *


######## Extract command line arguments


# Extract container list from command line arguments

con_list = sys.argv[1:5]

# Set path variables

log_data_path = sys.argv[5]

model_path = sys.argv[6]


######## Create log dataframe


con_dict = generate_con_dict(con_list)

log_df = generate_log_df_file(log_data_path, con_dict)

latest_logs = preprocess_log_df(log_df)


######## Predict the request mix of the next 250 requests


# Import best LSTM model

LSTM_model_p = open(model_path + "LSTM_model.pkl", "rb")
LSTM_model = pickle.load(LSTM_model_p)

# Predict requests

next_n_request_mix = LSTM_model.predict(latest_logs)


######## Predict request response time given predicted requests and available resources


# Prepare modelling table - add container information

request_resource_matrix = np.insert(
    next_n_request_mix,
    0,
    np.array([v for k, v in con_dict.items()]),
).reshape(1, -1)

# Import best supervised model

super_model_p = open(
    model_path + "RF_super_model.pkl", "rb"
)
super_model = pickle.load(super_model_p)


######## Predict request response time based on current resources


request_response_time = super_model.predict(
    request_resource_matrix
)

print(
    "Predicted average request response time with no changes: "
    + str(request_response_time)
    + " seconds"
)


######## Predict requests based on all resource permuations


permutations = list()

for combo in itertools.permutations([1, 2, 3, 4, 5, 6], 4):
    permutations.append(combo)

request_df = pd.DataFrame(next_n_request_mix)
request_repeat_df = pd.concat(
    [request_df] * 360, ignore_index=True
)
con_df = pd.DataFrame(
    permutations,
    columns=["app_1", "app_2", "app_3", "app_4"],
)
modelling_df = pd.concat(
    [con_df, request_repeat_df], axis=1, sort=False
)
modelling_df.iloc[:, 0:4] = MinMaxScaler().fit_transform(
    modelling_df.iloc[:, 0:4]
)

pred_response_time = super_model.predict(modelling_df)

modelling_df["request_response_time"] = pred_response_time

print("Best resource allocations:")

print(
    modelling_df.sort_values(by=["request_response_time"])[
        [
            "app_1",
            "app_2",
            "app_3",
            "app_4",
            "request_response_time",
        ]
    ]
    .iloc[0:50]
    .mean()
)
