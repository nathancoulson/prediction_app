import pandas as pd
import os
import re
import numpy as np
from os import listdir
from os.path import isfile, join
import datetime

def format_datetime(datetime):
    """
    Format datetime string into format which is easily parsed by Pandas
    """

    day = datetime[1:3]
    month = "09"
    year = "2019"
    time = datetime.split(":", 1)[1]

    formatted = year + "-" + month + "-" + day + " " + time

    return formatted


def clean_log(log, con_dict):
    """
    Parse line of log and return as dictionary
    """

    log_dict = dict()
    bits = log.split()

    formatted = format_datetime(
        re.sub('["' "]", "", bits[-8])
    )

    log_dict = {
        "resp_time": re.sub('["' "]", "", bits[-1]),
        "bytes_sent": re.sub('["' "]", "", bits[-2]),
        "resp_code": re.sub('["' "]", "", bits[-3]),
        "url": re.sub('["' "]", "", bits[-5]),
        "datetime": formatted,
    }

    event_dict = {**log_dict, **con_dict}

    return event_dict


def clean_error(error, con_dict):
    """
    Parse line of log of type error and return as dictionary
    """

    error_dict = dict()
    bits = error.split()

    error_dict = {
        "resp_time": np.nan,
        "bytes_sent": np.nan,
        "resp_code": re.sub("[(:]", "", bits[10]),
        "url": re.sub('["' "]", "", bits[-6]),
        "datetime": re.sub("[/]", "-", bits[2])
        + " "
        + bits[3],
    }

    event_dict = {**error_dict, **con_dict}

    return event_dict


def extract_con_info(filename):
    """
    Extract container information from log file name and return as dictionary
    """

    con_dict = dict()
    con_string = filename.split("4C", 1)[1].split(".", 1)[0]
    container_list = con_string.split("X")

    con_dict = {
        "app_1_containers": int(
            container_list[0].split("-")[1]
        ),
        "app_2_containers": int(
            container_list[1].split("-")[1]
        ),
        "app_3_containers": int(
            container_list[2].split("-")[1]
        ),
        "app_4_containers": int(
            container_list[3].split("-")[1]
        ),
    }

    return con_dict


def generate_log_df(log_files_path):
    """
    Function to take a path to a folder containing log files and generate a cleaned dataframe. Depends on the filenames being in unicode order.
    """

    all_logs = list()

    files = [
        f
        for f in listdir(log_files_path)
        if isfile(join(log_files_path, f)) and "logs" in f
    ]

    files.sort()

    for file in files:
        # Extract container information from filename
        container_dict = extract_con_info(file)

        with open((log_files_path + "{}").format(file), "r") as f:
            line = f.readline()
            while line:
                line = f.readline()
                if len(line.split()) == 13:
                    all_logs.append(
                        clean_log(line, container_dict)
                    )
                elif len(line.split()) == 32:
                    all_logs.append(
                        clean_error(line, container_dict)
                    )
            print(file + " done!")

    logs_df = pd.DataFrame(all_logs)

    return logs_df


def preprocess_log_df(log_df):
    # Create categorical variables for url requests

    log_df = pd.get_dummies(log_df, columns=["url"])

    # Create url request only dataframe

    req_df = log_df[
        [col for col in log_df.columns if "url" in col]
    ]

    logs = (
        req_df.tail(1000).to_numpy().reshape(1, 1000, 129)
    )

    return logs
