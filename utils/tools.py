from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold

import torch

from model.FFNN import FeedForwardNN
from utils.ml_dataset import xrDataset, xrDatasetPrediction

import xarray as xr



def _to_label_name(label_cols) -> List[str]:
    if isinstance(label_cols, int):
        label_cols = np.arange(1, label_cols + 1)
    elif isinstance(label_cols, list):
        label_cols = np.array(label_cols)
    label_names = [f"lead_{item}" for item in label_cols]
    return label_names

def _select_feature(
        df: pd.DataFrame,
        rule: str = 'all',
        custom_columns: List[str] = None,
        regex_columns: str = None) -> pd.DataFrame:
    if rule == 'all':
        selected_df = df.copy()
    elif rule == 'regex' and regex_columns is not None:
        selected_df = df.filter(regex=regex_columns)
    elif rule == 'custom' and custom_columns is not None:
        selected_df = df.filter(items=custom_columns)
    else:
        raise ValueError("Invalid rule or custom_columns or regex_columns.")

    return selected_df

def _0_1_normalize(df: [pd.DataFrame, xr.DataArray], dim: list = None):
    if isinstance(df, pd.DataFrame):
        df_max, df_min = df.max(), df.min()
    elif isinstance(df, xr.DataArray):
        df_max, df_min = df.max(dim=dim), df.min(dim=dim)
        print(f"da Max={df_max}, min={df_min}")
    norm = (df - df_min) / (df_max - df_min)
    return norm

def _reshape_xr(da, new_coord: str, reshape_coords: list) -> xr.DataArray:
    reshape_dict = {new_coord: reshape_coords}
    da_reshape = da.stack(**reshape_dict).drop_vars(reshape_coords)
    dimSize = da_reshape[new_coord].size
    da_reshape[new_coord] = np.arange(dimSize)
    return da_reshape


def _preprocess_X_y_data(
        features, labels,
        periods, label_cols,
        feature_selection_dict,
        X_dimension_dict,
):

    if isinstance(features, (xr.DataArray, xr.Dataset)):

        features = features["X"]
        features = xr.concat(
            [features.sel(time=slice(start, end)) for start, end in periods],
            dim="time"
        )
        features = features.dropna(dim="time", how="any")
        features = features.sel(**X_dimension_dict["select_features"])
        time_coords = features.coords["time"].data

    elif isinstance(features, pd.DataFrame):
        features = pd.concat(
            [features.loc[start:end] for start, end in periods]
        )
        features = features.dropna(how="any")
        time_coords = features.index

        features = _select_feature(
            df=features, rule=feature_selection_dict["rule"],
            custom_columns=feature_selection_dict["custom_columns"],
            regex_columns=feature_selection_dict["regex_columns"]
        )

    # process y labels
    if isinstance(labels, (xr.DataArray, xr.Dataset)):
        labels = labels["y"]
        labels = labels.sel(time=time_coords, lead=label_cols)
    elif isinstance(labels, pd.DataFrame):
        labels = labels[label_cols]
        time_coords = pd.DatetimeIndex(time_coords).strftime("%Y-%m-%d %H:%M:%S")
        labels = labels.loc[time_coords]


    if isinstance(features, (xr.DataArray, xr.Dataset)):
        if X_dimension_dict["reshape"] == "1D":
            features = _reshape_xr(
                features,
                new_coord=X_dimension_dict["new_coord"],
                reshape_coords=X_dimension_dict["reshape_coords"]
            )
    return features, labels


def _split_train_test(
        features, labels, split_method, n_splits,
        test_size, random_state, shuffle=True
):
    if split_method == "holdout":
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels,
            test_size=test_size,
            random_state=random_state, shuffle=shuffle
        )
        print(f"X_train shape: {X_train.shape}")
        if isinstance(X_train, (pd.DataFrame, pd.Series)):
            print(f"{X_train.describe()}\nTime: {X_train.index}")
            print(f"y_train: {y_train.describe()}")
        elif isinstance(X_train, (xr.DataArray, xr.Dataset)):
            print(f"X_train shape: {X_train.to_pandas().describe()}")
            print(f"X_train time: {X_train.coords['time']}")
        return [X_train, X_val, y_train, y_val]

    if split_method == "KFold":
        kfold = KFold(n_splits=n_splits, shuffle=shuffle)
        return kfold


def _get_train_modules_4_xr(
        X,
        hidden_size=None,
        output_size=None):
    features_num = X.shape[-1]
    labels_num = output_size

    print(f"Model input: {features_num} features")
    print(f"Model output: {labels_num} labels")

    device = torch.device("cpu")
    model = FeedForwardNN(
        input_size=features_num,
        hidden_size=hidden_size,
        output_size=labels_num
    )

    model.to(device)
    return model, device

def _get_dataloader(
        X, y=None,
        shuffle=None, batch_size=None
):
    if shuffle is None:
        shuffle = False
    X = X.values.astype(np.float32)
    if y is not None:
        y = y.values.astype(np.float32)
        dataset = xrDataset(X, y)
    elif y is None:
        dataset = xrDatasetPrediction(X)

    data_loader = dataset.create_dataloader(
        shuffle=shuffle,
        batch_size=batch_size
    )

    return data_loader