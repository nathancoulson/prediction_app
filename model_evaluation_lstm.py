import pandas as pd
import os
import sys
import numpy as np
import datetime
from numpy import array
import re


# LSTM Eval transformation functions

def most_common(lst):
    return max(set(lst), key=lst.count)


def reveal_bias(url):
    """
    Parse url string to count number of requests to each microservice
    """

    path = url.split("/", 1)[1]
    apps = re.sub("[/-]", " ", path).split()[:-1]

    return most_common(apps)


def add_bias_labels(col_lst):
    """
    Re-label columns to include microservice bias of request
    """

    for i in range(len(col_lst)):
        col_lst[i] = (
            col_lst[i] + "-bias-" + reveal_bias(col_lst[i])
        )

    return col_lst


def create_reqset_dict(col_list, req_list):
    reqset_dict = dict(
        zip(add_bias_labels(col_list), req_list)
    )

    return reqset_dict


def get_reqset_bias(dictionary):
    """
    Count number of requests for each app bias
    """

    bias_2 = sum(
        [v for k, v in dictionary.items() if "-bias-2" in k]
    )
    bias_3 = sum(
        [v for k, v in dictionary.items() if "-bias-3" in k]
    )
    bias_4 = sum(
        [v for k, v in dictionary.items() if "-bias-4" in k]
    )

    return (bias_2, bias_3, bias_4)


def get_app_bias_error(X_test, y_test, model, cumsum_cols):
    """
    Wrapper function to get app bias error for a given model
    """

    error_vectors_app_bias = list()

    for i in range(len(X_test)):
        predict_dict = create_reqset_dict(
            cumsum_cols,
            model.predict(X_test[i].reshape(1, 1000, 129))[
                0
            ],
        )
        predict_bias = get_reqset_bias(predict_dict)

        actual_dict = create_reqset_dict(
            cumsum_cols, y_test[i]
        )
        actual_bias = get_reqset_bias(actual_dict)

        error_vectors_app_bias.append(
            np.absolute(
                np.array(actual_bias)
                - np.array(predict_bias)
            )
        )

    return error_vectors_app_bias

def is_bias(url, lst):
    if url in lst:
        return True
    else:
        return False
