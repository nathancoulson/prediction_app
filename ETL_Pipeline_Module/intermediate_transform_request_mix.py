import pandas as pd
import os
import sys
import numpy as np
import datetime
from random import randint
from numpy import array
import pickle

def generate_train_val_test(train_df, test_df, path):
    '''
    Function to prepare output and inputs for LSTM model. Train, validation and test sets.
    '''

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

    with open(path + 'X_val.pkl', 'wb') as f: 
        pickle.dump(X_val, f)
    with open(path + 'y_val.pkl', 'wb') as f: 
        pickle.dump(sub_y_val, f)
    with open(path + 'X_train.pkl', 'wb') as f: 
        pickle.dump(X, f)
    with open(path + 'y_train.pkl', 'wb') as f: 
        pickle.dump(sub_y, f)
    with open(path + 'X_test.pkl', 'wb') as f: 
        pickle.dump(X_test, f)
    with open(path + 'y_test.pkl', 'wb') as f: 
        pickle.dump(sub_y_test, f)
        
    print("Train, validation and test sets prepared for LSTM model. Find them here: " + path)
