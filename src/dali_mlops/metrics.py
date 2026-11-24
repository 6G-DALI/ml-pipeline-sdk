"""Metric names and values in the Data Space vocabulary."""

import math
from numbers import Integral, Real
import re

ALIASES = {
    "accuracy": "model-accuracy",
    "f1": "model-f1",
    "f1-score": "model-f1",
    "rmse": "model-rmse",
    "mae": "model-mae",
    "r2": "model-r2",
    "latency-ms": "inference-latency-ms",
    "inference-latency-ms": "inference-latency-ms",
    "model-size-mb": "model-size-mb",
}


def metric_name(name):
    if not isinstance(name, str):
        raise ValueError("Metric names must be nonempty strings.")
    kebab = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not kebab:
        raise ValueError("Metric names must be nonempty strings.")
    return ALIASES.get(kebab, kebab)


def normalize_metrics(metrics, *, history=False):
    """Map names to Data Space names. With history=True an integer epoch is kept as is."""
    result = {}
    for name, value in metrics.items():
        if history and name == "epoch":
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError("The epoch of a metric history entry must be a nonnegative integer.")
            result["epoch"] = int(value)
            continue
        original, name = name, metric_name(name)
        if name in result:
            raise ValueError(f"Duplicate metric name after mapping: {original!r} maps to {name!r}.")
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError("Metric values must be finite numbers.")
        result[name] = float(value)
    return result


def normalize_history(history):
    return [normalize_metrics(point, history=True) for point in history]
