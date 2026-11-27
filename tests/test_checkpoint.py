import pytest

from dali_mlops import Checkpoint, SDKError, dump_joblib, dump_torch_state, load_joblib, load_torch_state


def test_joblib_roundtrip():
    checkpoint = dump_joblib({"trees": [1, 2, 3]})
    assert (checkpoint.filename, checkpoint.media_type) == ("checkpoint.joblib", "application/octet-stream")
    assert load_joblib(checkpoint) == {"trees": [1, 2, 3]}
    assert dump_joblib(1, "forest.pkl").filename == "forest.pkl"


def test_torch_state_roundtrip():
    torch = pytest.importorskip("torch")
    state = {"w": torch.arange(4.0), "epoch": 3}
    checkpoint = dump_torch_state(state)
    assert checkpoint.filename == "checkpoint.pt"
    loaded = load_torch_state(checkpoint)
    assert loaded["epoch"] == 3 and torch.equal(loaded["w"], state["w"])


@pytest.mark.parametrize("load", [load_joblib, load_torch_state])
def test_unreadable_checkpoints_say_so(load):
    pytest.importorskip("torch")
    with pytest.raises(SDKError, match="could not be loaded with the packages of this run"):
        load(Checkpoint.from_bytes(b"not a checkpoint"))
