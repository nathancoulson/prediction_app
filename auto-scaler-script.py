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

# Extract container list from command line arguments

con_list = sys.argv[1:]

# Data extraction pipeline

def format_datetime(datetime):
    # Bad hardcoded solution! Generalise later if needed
    
    day = datetime[1:3]
    month = "09"
    year = "2019"
    time = datetime.split(':', 1)[1]
    
    formatted = year + "-" + month + "-" + day + " " + time
    
    return formatted

def clean_log(log, con_dict):
    log_dict = dict()
    bits = log.split()
    
    formatted = format_datetime(re.sub('["'']', '',bits[-8]))

    log_dict = {
        "resp_time": re.sub('["'']', '',bits[-1]),
        "bytes_sent": re.sub('["'']', '',bits[-2]),
        "resp_code": re.sub('["'']', '',bits[-3]),
        "url": re.sub('["'']', '',bits[-5]),
        "datetime": formatted
    }
    
    event_dict = {**log_dict, **con_dict}
    
    return event_dict

def clean_error(error, con_dict):
    error_dict = dict()
    bits = error.split()
    
    error_dict = {
        "resp_time": np.nan,
        "bytes_sent": np.nan,
        "resp_code": re.sub('[(:]', '',bits[10]),
        "url": re.sub('["'']', '',bits[-6]),
        "datetime": re.sub('[/]', '-',bits[2]) + " " + bits[3]
    }
    
    event_dict = {**error_dict, **con_dict}
    
    return event_dict

def apply_minmax(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

def generate_con_dict(con_list):
    con_dict = dict()
    
    con_dict = {
        "app_1_containers": apply_minmax(int(con_list[0]), 1, 6),
        "app_2_containers": apply_minmax(int(con_list[1]), 1, 6),
        "app_3_containers": apply_minmax(int(con_list[2]), 1, 6),
        "app_4_containers": apply_minmax(int(con_list[3]), 1, 6)
    }
    
    return con_dict

def generate_log_df_file(log_file_path, container_dict):
    all_logs = list()
    with open(log_file_path, "r") as f:
        line = f.readline()
        while line:
            line = f.readline()
            if len(line.split()) == 13:
                all_logs.append(clean_log(line, container_dict))
            elif len(line.split()) == 32:
                all_logs.append(clean_error(line, container_dict))
                
    log_df = pd.DataFrame(all_logs)
        
    return log_df

def preprocess_log_df(log_df):
    # Create categorical variables for url requests

    log_df = pd.get_dummies(log_df, columns = ['url'])

    # Create url request only dataframe

    req_df = log_df[[col for col in log_df.columns if "url" in col]]

    logs = req_df.tail(1000).to_numpy().reshape(1,1000,129)
    
    return logs



log_data_path = "../Data/"

model_path = "./Models/"

# Create URL one-hot-encoded dataframe

print("con_list")
print(con_list)

con_dict = generate_con_dict(con_list)

log_df = generate_log_df_file(log_data_path + 'w-logs-11-09-19-2_3_4bias4C1-2X2-6X3-2X4-6.txt', con_dict)

latest_logs = preprocess_log_df(log_df)

print(latest_logs)

######## Predict the request mix of the next 250 requests

# Import best LSTM model

LSTM_model_p = open(model_path + 'LSTM_model.pkl', 'rb')
LSTM_model = pickle.load(LSTM_model_p)

# Predict requests

next_n_request_mix = LSTM_model.predict(latest_logs)

print(next_n_request_mix)

with open('next_n_request_mix.pkl', 'wb') as f:
    pickle.dump(next_n_request_mix, f)

######## Predict request response time given predicted requests and available resources

# Prepare modelling table - add container information

request_resource_matrix = np.insert(next_n_request_mix, 0, np.array([v for k,v in con_dict.items()])).reshape(1,-1)

# Import best supervised model

super_model_p = open(model_path + 'super_model.pkl', 'rb')
super_model = pickle.load(super_model_p)

# Predict requests based on current resources

request_response_time = super_model.predict(request_resource_matrix)

print("Predicted average request response time with no changes: " + str(request_response_time) + " seconds")

# Predict requests based on all resoure permuations

permutations = list ()

for combo in itertools.permutations([1,2,3,4,5,6], 4): 
    permutations.append(combo)

request_df = pd.DataFrame(next_n_request_mix)
request_repeat_df = pd.concat([request_df]*360, ignore_index=True)
con_df = pd.DataFrame(permutations, columns = ["app_1", "app_2", "app_3", "app_4"])
modelling_df = pd.concat([con_df, request_repeat_df], axis=1, sort=False)
modelling_df.iloc[:,0:4] = MinMaxScaler().fit_transform(modelling_df.iloc[:,0:4])

pred_response_time = super_model.predict(modelling_df)

modelling_df["request_response_time"] = pred_response_time

print("Best resource allocations:")

print(modelling_df.sort_values(by=["request_response_time"])[["app_1", "app_2", "app_3", "app_4", "request_response_time"]].iloc[0:10])
