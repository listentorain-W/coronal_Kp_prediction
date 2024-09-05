from typing import Dict, List
import logging

import numpy as np
import pandas as pd
import xarray as xr

import torch

from utils.tools import \
    _0_1_normalize, _reshape_xr, _select_feature, \
    _to_label_name, _get_dataloader, _get_train_modules_4_xr


log = logging.getLogger(__name__)


def _add_coords(da, coords: Dict):
    for k, v in coords.items():
        da = da.assign_coords(**{k: v})
    return da


def _wrap_xr_dataset(data: np.ndarray, dims: tuple, coords: Dict) -> xr.DataArray:
    return xr.DataArray(data, dims=dims, coords=coords)



@torch.no_grad()
def _predict_step(model, device, data_loader):
    model.eval()
    predictions = []
    for idx, features in enumerate(data_loader):
        features = features.to(device)
        y_pred = model(features)
        predictions.extend(y_pred.cpu().numpy())
    return np.array(predictions)


def _preprocess_data_4_xr(
        features,
        periods,
        feature_selection_dict: Dict,
        X_dimension_dict: Dict
):
    # process X features
    if isinstance(features, (xr.DataArray, xr.Dataset)):
        features = features["X"]
        features = xr.concat(
            [features.sel(time=slice(start, end)) for start, end in periods],
            dim="time"
        )
        features = features.dropna(dim="time", how="any")

        features = features.sel(**X_dimension_dict["select_features"])

        time_coords = features.coords["time"].data
        print(f"X data shape: {features.sizes}")

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

    if isinstance(features, (xr.DataArray, xr.Dataset)):
        if X_dimension_dict["reshape"] == "1D":
            features = _reshape_xr(
                features,
                new_coord=X_dimension_dict["new_coord"],
                reshape_coords=X_dimension_dict["reshape_coords"]
            )

    return features, time_coords


def predict_model_4_xr(
        model, device,
        features,
        config: Dict
):
    print("Predicting model...")

    # get dataloader
    real_dataloader = _get_dataloader(
        X=features, y=None,
        shuffle=False, batch_size=config["batch_size"]
    )

    predict_result = _predict_step(
        model=model, device=device,
        data_loader=real_dataloader
    )

    return predict_result


def predict_multi_lead_with_K_fold(
        tmp: Dict,
        features,
        parameters: Dict, dataset_params: Dict
):
    label_cols = _to_label_name(parameters["label_cols"])

    features, X_time_coords = _preprocess_data_4_xr(
        features=features,
        periods=dataset_params["predict_periods"],
        feature_selection_dict=dataset_params["feature_selection_dict"],
        X_dimension_dict=dataset_params["X_dimension_dict"]
    )
    # get model and device
    model, device = _get_train_modules_4_xr(
        X=features,
        hidden_size=dataset_params["hidden_size"],
        output_size=dataset_params["output_size"]
    )


    fold_list = []
    for fold in range(dataset_params["n_splits"]):
        lead_list = []
        try:
            EXP_STAMP = dataset_params["date_stamp"]
        except KeyError:
            EXP_STAMP = parameters["date_stamp"]
        # load torch model from save_path
        save_path = f"{dataset_params['model_save_path']}/multi_lead_{EXP_STAMP}_fold{fold}.pth"

        model_dict = torch.load(save_path, map_location=device)

        for lead, label_col in enumerate(label_cols, 1):
            print(f"Predicting {label_col}...")

            model.load_state_dict(model_dict[label_col])
            predict_result = predict_model_4_xr(
                model=model, device=device,
                features=features,
                config=parameters
            )

            predict_result = _wrap_xr_dataset(
                predict_result[:, 0], dims=("time",),
                coords={"time": X_time_coords}
            )
            predict_result = _add_coords(
                predict_result, coords={"lead": lead}
            )
            lead_list.append(predict_result)

        lead_pred = xr.concat(lead_list, dim="lead")
        lead_pred = _add_coords(
            lead_pred, coords={"fold": fold}
        )
        fold_list.append(lead_pred)

    prediction = xr.concat(fold_list, dim="fold")
    prediction.to_dataset(name="prediction")
    print(f"Predict result: {prediction}, "
          f"Max: {prediction.max(dim=['time', 'lead'])}, "
          f"Min: {prediction.min(dim=['time', 'lead'])}")
    return prediction


if __name__ == "__main__":
    raise NotImplementedError("This script is not intended to be executed directly.")

