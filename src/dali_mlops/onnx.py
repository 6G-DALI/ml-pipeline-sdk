"""Optional scikit-learn export support. Publication accepts any model format."""

from .types import ExportedModel


def export_sklearn_onnx(model, *, input_name: str, feature_count: int, filename: str = "model.onnx") -> ExportedModel:
    """Convert a fitted estimator with a named [batch, feature_count] float32 input."""
    import onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    if not input_name or feature_count <= 0 or getattr(model, "n_features_in_", None) != feature_count:
        raise ValueError("The ONNX input schema must match the fitted estimator's feature count.")
    graph = convert_sklearn(model, initial_types=[(input_name, FloatTensorType([None, feature_count]))])
    onnx.checker.check_model(graph)
    tensor = graph.graph.input[0].type.tensor_type
    if tensor.shape.dim[1].dim_value != feature_count or tensor.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("The converted graph does not match the float32 input schema.")
    return ExportedModel.from_bytes(graph.SerializeToString(), filename, "application/onnx")
