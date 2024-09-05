from conf.parameters import data_config, multi_model_training
from typing import Dict, List
import pandas as pd
from predicting import predict_multi_lead_with_K_fold


def predict_results():
    X = pd.read_csv(
        "X_test.csv",
        index_col=["time"],
        parse_dates=True
    )
    tmp = {}
    output = predict_multi_lead_with_K_fold(
        tmp=tmp, features=X,
        parameters=multi_model_training, dataset_params=data_config
    )
    return output


if __name__ == '__main__':
    forecast_results = predict_results()
