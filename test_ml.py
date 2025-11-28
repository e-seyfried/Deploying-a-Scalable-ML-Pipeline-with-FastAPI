import os

import numpy as np
import pandas as pd

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# Same categorical features as in train_model.py
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def _load_sample_data(n_rows=200):
    """
    Helper function to load a small sample of census data
    for use in tests.
    """
    project_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_path, "data", "census.csv")
    data = pd.read_csv(data_path)
    return data.head(n_rows)


def test_process_data_outputs_shapes():
    """
    process_data should return X and y with matching, non-zero lengths.
    """
    data = _load_sample_data()
    X, y, _, _ = process_data(
        data,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )

    assert X.shape[0] == y.shape[0]
    assert X.shape[0] > 0


def test_train_model_and_inference_predictions():
    """
    train_model should produce a model that can make binary predictions
    with the same length as the input labels.
    """
    data = _load_sample_data()
    X, y, _, _ = process_data(
        data,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )

    model = train_model(X, y)
    preds = inference(model, X)

    assert len(preds) == len(y)
    # Predictions should be binary (0/1)
    unique_preds = np.unique(preds)
    assert set(unique_preds).issubset({0, 1})


def test_compute_model_metrics_known_values():
    """
    compute_model_metrics should return correct precision, recall, and F1
    for a simple, known example.
    """
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    precision, recall, f1 = compute_model_metrics(y_true, y_pred)

    # For this pattern, all three should be 0.5
    assert abs(precision - 0.5) < 1e-6
    assert abs(recall - 0.5) < 1e-6
    assert abs(f1 - 0.5) < 1e-6
