import pandas as pd
from ETL_Pipeline_Module.data_extraction import (
    clean_log,
    clean_error,
)


def apply_minmax(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def generate_con_dict(con_list):
    """
    Create dictionary of container allocations from command line input
    """

    con_dict = dict()

    con_dict = {
        "app_1_containers": apply_minmax(
            int(con_list[0]), 1, 6
        ),
        "app_2_containers": apply_minmax(
            int(con_list[1]), 1, 6
        ),
        "app_3_containers": apply_minmax(
            int(con_list[2]), 1, 6
        ),
        "app_4_containers": apply_minmax(
            int(con_list[3]), 1, 6
        ),
    }

    return con_dict


def generate_log_df_file(log_file_path, container_dict):
    """
    Generate dataframe from single log file - modified version of multi-file function in data extraction
    """

    all_logs = list()
    with open(log_file_path, "r") as f:
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

    log_df = pd.DataFrame(all_logs)

    return log_df


def preprocess_log_df(log_df):
    """
    Select last 1000 logs and reshape for LSTM modelling
    """

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
