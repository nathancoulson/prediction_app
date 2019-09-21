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


path = '../Datasets/'
model_path = '../Models/'

# Read in config file - column list

cumsum_cols_p = open(path + 'cumsum_cols.pkl', 'rb')
cumsum_cols = pickle.load(cumsum_cols_p)

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
        predict_dict = create_reqset_dict(cumsum_cols, model.predict(X_test[i].reshape(1,1000,129))[0])
        predict_bias = get_reqset_bias(predict_dict)

        actual_dict = create_reqset_dict(cumsum_cols, y_test[i])
        actual_bias = get_reqset_bias(actual_dict)

        error_vectors_app_bias.append(np.absolute(np.array(actual_bias) - np.array(predict_bias)))

    return error_vectors_app_bias

# Read in prepared datasets

X_val_p = open(path + 'X_val.pkl', 'rb')
X_val = pickle.load(X_val_p)

sub_y_val_p = open(path + 'y_val.pkl', 'rb')
sub_y_val = pickle.load(sub_y_val_p)

X_train_p = open(path + 'X_train.pkl', 'rb')
X = pickle.load(X_train_p)

y_train_p = open(path + 'y_train.pkl', 'rb')
sub_y = pickle.load(y_train_p)

X_test_p = open(path + 'X_test.pkl', 'rb')
X_test = pickle.load(X_test_p)

y_test_p = open(path + 'y_test.pkl', 'rb')
sub_y_test = pickle.load(y_test_p)

results = list()

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)

# grid search

optimizers = [keras.optimizers.Adam(lr=0.01)]
epochs = [1]
batches = [128]
num_units = [50,75,125]
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
                        "app-bias-mean-holdout-set": app_bias_mean,
                        "model-object": model
                    }
                    
                    print(model_dict)
                    
                    results.append(model_dict)

                    with open('../LSTM_new_results/results_new_21_128.pkl', 'wb') as f:
                        pickle.dump(results, f)
