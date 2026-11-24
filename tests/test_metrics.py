import pytest

from dali_mlops.metrics import metric_name, normalize_history, normalize_metrics


@pytest.mark.parametrize(("alias", "expected"), [
    ("accuracy", "model-accuracy"), ("f1", "model-f1"), ("f1_score", "model-f1"), ("F1-Score", "model-f1"),
    ("rmse", "model-rmse"), ("mae", "model-mae"), ("r2", "model-r2"),
    ("latency_ms", "inference-latency-ms"), ("inference_latency_ms", "inference-latency-ms"),
    ("model_size_mb", "model-size-mb"), ("model-rmse", "model-rmse"),
    ("Train Loss", "train-loss"), ("val__loss", "val-loss"),
])
def test_names(alias, expected):
    assert metric_name(alias) == expected


def test_values_become_floats():
    assert normalize_metrics({"mae": 1, "r2": 0.5}) == {"model-mae": 1.0, "model-r2": 0.5}


@pytest.mark.parametrize("metrics", [
    {"rmse": 1, "model-rmse": 2}, {"f1": 1, "f1_score": 2}, {"x": float("nan")}, {"x": True}, {"x": "1"},
    {"": 1}, {"%%": 1}, {1: 1},
])
def test_invalid(metrics):
    with pytest.raises(ValueError):
        normalize_metrics(metrics)


def test_history_keeps_integer_epoch():
    assert normalize_history([{"epoch": 0, "accuracy": 0.5}, {"epoch": 1, "accuracy": 0.7, "loss": 1}]) == [
        {"epoch": 0, "model-accuracy": 0.5}, {"epoch": 1, "model-accuracy": 0.7, "loss": 1.0},
    ]


@pytest.mark.parametrize("epoch", [1.5, -1, True, "1"])
def test_history_epoch_must_be_a_nonnegative_integer(epoch):
    with pytest.raises(ValueError):
        normalize_history([{"epoch": epoch}])
