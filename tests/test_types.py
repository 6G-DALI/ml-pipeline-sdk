import pytest

from dali_mlops import ExportedModel


def test_exported_model(tmp_path):
    path = tmp_path / "model.bin"
    path.write_bytes(b"model")
    assert ExportedModel.from_file(path, "application/octet-stream").open().read() == b"model"
    with pytest.raises(ValueError):
        ExportedModel.from_bytes(b"model", "../escape", "application/octet-stream")
