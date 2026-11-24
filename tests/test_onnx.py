import numpy as np
import onnx
import pytest
from sklearn.linear_model import LinearRegression

from dali_mlops.onnx import export_sklearn_onnx, export_torch_onnx


def test_sklearn_export_defaults_to_model_onnx():
    estimator = LinearRegression().fit(np.arange(6, dtype=np.float32).reshape(3, 2), [1, 2, 3])
    exported = export_sklearn_onnx(estimator, input_name="x", feature_count=2)
    assert (exported.filename, exported.media_type) == ("model.onnx", "application/onnx")
    onnx.checker.check_model(onnx.load_model_from_string(exported.data))


def test_sklearn_export_checks_feature_count():
    estimator = LinearRegression().fit(np.arange(6, dtype=np.float32).reshape(3, 2), [1, 2, 3])
    with pytest.raises(ValueError):
        export_sklearn_onnx(estimator, input_name="x", feature_count=3)


def test_torch_export_needs_the_extra(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    with pytest.raises(ImportError, match="torch"):
        export_torch_onnx(object(), None)


def test_torch_export():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 1)
    exported = export_torch_onnx(model, torch.zeros(1, 3), input_name="features")
    graph = onnx.load_model_from_string(exported.data)
    assert graph.graph.input[0].name == "features"
    assert graph.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch"
