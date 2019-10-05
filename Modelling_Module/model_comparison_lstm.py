import pandas as pd
import re


def create_lstm_model_df(all_results_list):
    """
    Create lstm meta model dataframe for comparison
    """

    all_results_dict = dict()

    for item in all_results_list:
        for k, v in item.items():
            model_params = k.split("_")
            all_results_dict[
                ("_".join(model_params[0:2]))
            ] = {
                model_params[3].split(":")[0]: int(
                    model_params[3].split(":")[1]
                ),
                model_params[4].split(":")[0]: int(
                    model_params[4].split(":")[1]
                ),
                model_params[5].split(":")[0]: int(
                    model_params[5].split(":")[1]
                ),
                model_params[6].split(":")[0]: int(
                    model_params[6].split(":")[1]
                ),
                "val_loss": float(
                    v["model-history"]["val_loss"][0]
                ),
                "loss": float(
                    v["model-history"]["loss"][0]
                ),
                "MAE-holdout-set": float(
                    v["MAE-holdout-set"].split()[1]
                ),
                "app-bias-mean-holdout-set": float(
                    v["app-bias-mean-holdout-set"]
                ),
            }

    all_results_df = pd.DataFrame(all_results_dict).T

    return all_results_df


def get_model(lst, datestamp):
    """
    Get model from model list by datestamp
    """

    for item in lst:
        for k, v in item.items():
            model_params = k.split("_")
            date_stamp = "_".join(model_params[0:2])
            if date_stamp == datestamp:
                return v["model-object"]
