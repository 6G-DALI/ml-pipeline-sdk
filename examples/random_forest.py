"""Forecast the next observed container CPU percentage from telemetry."""

from typing import Annotated

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from zenml import step

from dali_mlops import ExportedModel, TrainingOutput, dali_pipeline
from dali_mlops.onnx import export_sklearn_onnx

FEATURE_NAMES = ["cpu_percent", "memory_used_mib", "pids"]
MEMORY_FACTORS = {
    "B": 1, "kB": 1000, "KB": 1000, "MB": 1000**2, "GB": 1000**3, "TB": 1000**4,
    "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4,
}


@step
def preprocess(
    df: pd.DataFrame, test_size: float,
) -> tuple[
    Annotated[np.ndarray, "train_x"], Annotated[np.ndarray, "train_y"],
    Annotated[np.ndarray, "test_x"], Annotated[np.ndarray, "test_y"],
]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one.")
    required = ["timestamp", "container_name", "cpu_percent", "memory_usage", "pids"]
    missing = set(required).difference(df.columns)
    if missing:
        raise ValueError(f"Missing telemetry columns: {sorted(missing)}")
    data = df[required].copy()
    if data.isna().any().any():
        raise ValueError("Required telemetry columns contain missing values.")
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    if data["timestamp"].isna().any() or data["container_name"].astype(str).str.strip().eq("").any():
        raise ValueError("Each observation needs a timestamp and a container name.")
    if data.duplicated(["container_name", "timestamp"]).any():
        raise ValueError("Expected one observation per container and timestamp.")
    data["cpu_percent"] = pd.to_numeric(data["cpu_percent"].astype(str).str.strip().str.removesuffix("%"))
    memory = data["memory_usage"].astype(str).str.split("/", n=1).str[0].str.extract(
        r"^\s*(\d+(?:\.\d+)?)\s*(B|kB|KB|MB|GB|TB|KiB|MiB|GiB|TiB)\s*$"
    )
    data["memory_used_mib"] = pd.to_numeric(memory[0]) * memory[1].map(MEMORY_FACTORS) / 1024**2
    data["pids"] = pd.to_numeric(data["pids"])
    features = data[FEATURE_NAMES].to_numpy(dtype=np.float32)
    if not np.isfinite(features).all() or (features < 0).any() or (data["pids"] % 1 != 0).any():
        raise ValueError("Telemetry must have finite nonnegative features, supported memory units and integer pids.")
    # Multi-core containers can exceed 100 percent CPU.
    data = data.sort_values(["container_name", "timestamp"])
    groups = data.groupby("container_name", sort=False)
    data["next_cpu_percent"] = groups["cpu_percent"].shift(-1)
    data["target_timestamp"] = groups["timestamp"].shift(-1)
    times = data["timestamp"].drop_duplicates().sort_values().tolist()
    split = int(len(times) * (1 - test_size))
    if not 0 < split < len(times):
        raise ValueError("Not enough distinct timestamps for the requested split.")
    cutoff = times[split]
    pairs = data.dropna(subset=["target_timestamp"])
    train_data = pairs.loc[pairs["target_timestamp"] < cutoff]
    test_data = pairs.loc[pairs["timestamp"] >= cutoff]
    if train_data.empty or len(test_data) < 2:
        raise ValueError("The chronological split needs training pairs and at least two test pairs for R².")
    return (
        train_data[FEATURE_NAMES].to_numpy(dtype=np.float32),
        train_data["next_cpu_percent"].to_numpy(dtype=np.float32),
        test_data[FEATURE_NAMES].to_numpy(dtype=np.float32),
        test_data["next_cpu_percent"].to_numpy(dtype=np.float32),
    )


@step
def train(
    train_x: np.ndarray, train_y: np.ndarray, n_estimators: int, max_depth: int | None,
    min_samples_split: int, min_samples_leaf: int, random_state: int,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf, random_state=random_state, n_jobs=-1,
    )
    return model.fit(train_x, train_y)


@step
def evaluate(model: RandomForestRegressor, test_x: np.ndarray, test_y: np.ndarray) -> dict[str, float]:
    predicted = model.predict(test_x)
    return {
        "mae": float(mean_absolute_error(test_y, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(test_y, predicted))),
        "r2": float(r2_score(test_y, predicted)),
    }


@step
def export(model: RandomForestRegressor) -> ExportedModel:
    return export_sklearn_onnx(model, input_name="telemetry", feature_count=len(FEATURE_NAMES))


@dali_pipeline(
    name="random_forest_pipeline", dataset="csv",
    requirements=["scikit-learn>=1.5,<2", "skl2onnx>=1.19,<2", "onnx>=1.17,<2"],
)
def random_forest_pipeline(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 50,
    max_depth: int | None = 10,
    min_samples_split: int = 2,
    min_samples_leaf: int = 2,
):
    train_x, train_y, test_x, test_y = preprocess(df, test_size)
    model = train(train_x, train_y, n_estimators, max_depth, min_samples_split, min_samples_leaf, random_state)
    return TrainingOutput(export(model), evaluate(model, test_x, test_y))
