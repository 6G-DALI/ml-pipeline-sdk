import pytest

from dali_mlops import ExportedModel


def test_exported_model(tmp_path):
    path = tmp_path / "model.bin"
    path.write_bytes(b"model")
    assert ExportedModel.from_file(path, "application/octet-stream").open().read() == b"model"
    with pytest.raises(ValueError):
        ExportedModel.from_bytes(b"model", "../escape", "application/octet-stream")


def test_defaults_to_onnx(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"model")
    assert ExportedModel.from_file(path).media_type == "application/onnx"
    exported = ExportedModel.from_bytes(b"model")
    assert (exported.filename, exported.media_type) == ("model.onnx", "application/onnx")


def test_checkpoint(tmp_path):
    from dali_mlops import Checkpoint

    path = tmp_path / "state.pt"
    path.write_bytes(b"state")
    assert Checkpoint.from_file(path).open().read() == b"state"
    assert Checkpoint.from_file(path).media_type == "application/octet-stream"
    assert Checkpoint("a.bin", data=b"x") == Checkpoint.from_bytes(b"x", "a.bin")
    assert Checkpoint.from_bytes(b"x").filename == "checkpoint.bin"
    for make in (
        lambda: Checkpoint.from_bytes(b"x", "../escape"), lambda: Checkpoint.from_bytes(b""),
        lambda: Checkpoint("a.bin"), lambda: Checkpoint.from_bytes(b"x", "a.bin", "nonsense"),
    ):
        with pytest.raises(ValueError):
            make()
