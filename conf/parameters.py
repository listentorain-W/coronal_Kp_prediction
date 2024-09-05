
data_config = {
    "date_stamp": "predict_model_names",
    "predict_periods": [["2010-01", "2023-12"]],

    "split_method": "KFold",
    "n_splits": 5,

    "model_save_path": '/your/model/paths',

    "X_dimension_dict":
        {
            "reshape": "0D",
            "reshape_coords": ["var", "lead", "pixel"],
            "new_coord": "features",
            "normalize_coords": ["time", "lead", "pixel"],
            "select_features": {},
        },

    "feature_selection_dict":
        {
            "rule": "all",
            "regex_columns": None,
            "custom_columns": None,
        },

    "hidden_size": 25,
    "output_size": 1
}

multi_model_training = {
    "test_size": 0.1,
    "batch_size": 1,

    "label_cols": 24,
    "train_data_shuffle": True,

    "learning_rate": 0.0035,
    "momentum": 0.125,
    "beta1": 0.9,
    "eps": 1e-08,

    "num_epochs": 8,
    "loss_function": "MSELoss",
    "optimizer": "Adam",
    "scheduler": "StepLR",
}
