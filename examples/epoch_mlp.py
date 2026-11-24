"""Train a small neural network and report per-epoch metrics."""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from zenml import step

from dali_mlops import ExportedModel, TrainingOutput, dali_pipeline
from dali_mlops.onnx import export_sklearn_onnx


@step
def fit(df: pd.DataFrame, target: str, epochs: int, learning_rate: float) -> tuple[ExportedModel, dict, list]:
    features = df.drop(columns=[target]).to_numpy(dtype=np.float32)
    labels = df[target].to_numpy(dtype=np.float32)
    model = MLPRegressor(hidden_layer_sizes=(8,), learning_rate_init=learning_rate, random_state=0)
    history = []
    for epoch in range(epochs):
        model.partial_fit(features, labels)
        predicted = model.predict(features)
        history.append({"epoch": epoch, "loss": float(model.loss_), "mae": float(np.abs(predicted - labels).mean())})
    exported = export_sklearn_onnx(model, input_name="features", feature_count=features.shape[1])
    return exported, {key: value for key, value in history[-1].items() if key != "epoch"}, history


@dali_pipeline(dataset="csv", requirements=["scikit-learn>=1.5,<2", "skl2onnx>=1.19,<2", "onnx>=1.17,<2"])
def epoch_mlp_pipeline(df: pd.DataFrame, target: str = "target", epochs: int = 10, learning_rate: float = 0.01):
    model, metrics, history = fit(df, target, epochs, learning_rate)
    return TrainingOutput(model, metrics, history)
