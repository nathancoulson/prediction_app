import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import keras
import datetime

from random import randint
from numpy import array
from numpy import argmax
from random import randint
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


from math import sin
from math import pi
from math import exp
from random import randint
from random import uniform

from numpy import array
from math import ceil
from math import log10
from keras.layers import TimeDistributed
from keras.layers import RepeatVector

from random import random
from numpy import cumsum
from numpy import array_equal
from keras.layers import Bidirectional

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import pickle

from sklearn.preprocessing import MinMaxScaler

import re

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

tf.keras.backend.set_session(session)

# Eval transformation functions

def most_common(lst): 
    return max(set(lst), key = lst.count)

def reveal_bias(url):
    path = url.split('/', 1)[1]
    apps = re.sub('[/-]', ' ',path).split()[:-1]
    return most_common(apps)

def add_bias_labels(col_lst):
    for i in range(len(col_lst)):
        col_lst[i] = col_lst[i] + "-bias-" + reveal_bias(col_lst[i])
    return col_lst

def create_reqset_dict(col_list, req_list):
    reqset_dict = dict(zip(add_bias_labels(col_list),req_list))
    return reqset_dict

def get_reqset_bias(dictionary):
    bias_2 = sum([v for k,v in dictionary.items() if "-bias-2" in k])
    bias_3 = sum([v for k,v in dictionary.items() if "-bias-3" in k])
    bias_4 = sum([v for k,v in dictionary.items() if "-bias-4" in k])
    
    return (bias_2, bias_3, bias_4)

def get_app_bias_error(X_test, y_test, model):
    error_vectors_app_bias = list()
    for i in range(len(X_test)):
        predict_dict = create_reqset_dict([col for col in test_df.columns if "CS_" in col], model.predict(X_test[i].reshape(1,1000,129))[0])
        predict_bias = get_reqset_bias(predict_dict)

        actual_dict = create_reqset_dict([col for col in test_df.columns if "CS_" in col], y_test[i])
        actual_bias = get_reqset_bias(actual_dict)

        error_vectors_app_bias.append(np.absolute(np.array(actual_bias) - np.array(predict_bias)))

    return error_vectors_app_bias

train_df = pd.read_parquet('train.parquet')
test_df = pd.read_parquet('test.parquet')

# First half of Test as validation - second half: holdout

holdout_df = test_df[:1120000]
test_df = test_df[1120000:2240000]

# Prepare test data as validation

y_val = array(test_df[[col for col in test_df.columns if "CS_" in col]]).reshape(1120000,129)

X_val = test_df[[col for col in test_df.columns if "CS_" not in col]].to_numpy()
X_val = X_val.reshape(1120,1000,129)

# Get the thousandth request vector for y

sub_y_val = y_val[::1000]

# Prepare training data

y = array(train_df[[col for col in train_df.columns if "CS_" in col]]).reshape(2241000,129)

X = train_df[[col for col in train_df.columns if "CS_" not in col]].to_numpy()
X = X.reshape(2241,1000,129)

# Get the thousandth request vector for y

sub_y = y[::1000]

# Prepare test data

y_test = array(holdout_df[[col for col in holdout_df.columns if "CS_" in col]]).reshape(1120000,129)

X_test = holdout_df[[col for col in holdout_df.columns if "CS_" not in col]].to_numpy()
X_test = X_test.reshape(1120,1000,129)

# Get the thousandth request vector for y

sub_y_test = y_test[::1000]

results = list()

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)

# grid search

optimizers = [keras.optimizers.Adam(lr=0.01), keras.optimizers.Adam(lr=0.001)]
epochs = [1]
batches = [128]
num_units = [75,125]
num_layers = [2,3,4]

for optimizer in optimizers:
    for epoch in epochs:
        for batch in batches:
            for num in num_units:
                for layers in num_layers:
                    
                    model_dict = dict()
                    
                    model = Sequential()
                    model.add(LSTM(num, return_sequences=True, input_shape=(1000, 129)))
                    if layers == 3:
                        model.add(LSTM(num, return_sequences=True))
                    elif layers == 4:
                        model.add(LSTM(num, return_sequences=True))
                        model.add(LSTM(num, return_sequences=True))
                    model.add(LSTM(num))
                    model.add(Dense(129))
                    model.compile(loss='mae', optimizer=optimizer)

                    # fit model
                    history = model.fit(X, sub_y, batch_size=batch, epochs=epoch, validation_data=(X_val, sub_y_val), shuffle=False)
                    
                    model_hash = "model" + "_".join(str(datetime.datetime.now()).split()) + "_optimizer:" + str(optimizer) + "_epochs:" + str(epoch) + "_batches:" + str(batch) + "_units:" + str(num) + "_layers:" + str(layers)
                    
                    # evaluate model by MAE

                    loss = model.evaluate(X_test, sub_y_test, verbose=0)
                    loss_string = 'MAE: %f' % loss

                    # compared predicted request set "app bias" with actual "app bias"

                    app_bias_list = get_app_bias_error(X_test, sub_y_test, model)

                    app_bias_df = pd.DataFrame(app_bias_list)

                    app_bias_mean = app_bias_df.mean().mean()
                    
                    # save model data in dictionary
                    
                    model_dict[model_hash] = {
                        "model-json": model.to_json(),
                        "model-history": history.history,
                        "MAE-holdout-set": loss_string,
                        "app-bias-mean-holdout-set": app_bias_mean
                    }
                    
                    print(model_dict)
                    
                    results.append(model_dict)

                    with open('results_reduced_19_5_001', 'wb') as f:
                        pickle.dump(results, f)
