"""Checkpoint helpers for common formats. The libraries are imported only when a helper runs."""

import io

from .types import Checkpoint, SDKError

LOAD_FAILED = "The checkpoint could not be loaded with the packages of this run."


def dump_joblib(obj, filename: str = "checkpoint.joblib") -> Checkpoint:
    """Serialize any joblib-picklable object (an sklearn estimator, a dict of state...)."""
    try:
        import joblib
    except ImportError:
        raise ImportError("dump_joblib needs joblib. Install scikit-learn or joblib in the pipeline requirements.") from None
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    return Checkpoint.from_bytes(buffer.getvalue(), filename)


def load_joblib(checkpoint: Checkpoint):
    """Load an object written by ``dump_joblib``. Loading runs pickle code: only use the platform's checkpoints."""
    try:
        import joblib
    except ImportError:
        raise ImportError("load_joblib needs joblib. Install scikit-learn or joblib in the pipeline requirements.") from None
    try:
        return joblib.load(checkpoint.open())
    except Exception as error:
        raise SDKError(f"{LOAD_FAILED} ({type(error).__name__})") from None


def dump_torch_state(state_dict, filename: str = "checkpoint.pt") -> Checkpoint:
    """Serialize a torch ``state_dict`` (or any dict of tensors and plain values). Needs the ``torch`` extra."""
    try:
        import torch
    except ImportError:
        raise ImportError("dump_torch_state needs torch. Install 6g-dali-mlops[torch].") from None
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return Checkpoint.from_bytes(buffer.getvalue(), filename)


def load_torch_state(checkpoint: Checkpoint, map_location="cpu"):
    """Load a ``state_dict`` written by ``dump_torch_state``. Tensors and plain containers only (weights_only)."""
    try:
        import torch
    except ImportError:
        raise ImportError("load_torch_state needs torch. Install 6g-dali-mlops[torch].") from None
    try:
        return torch.load(checkpoint.open(), map_location=map_location, weights_only=True)
    except Exception as error:
        raise SDKError(f"{LOAD_FAILED} ({type(error).__name__})") from None
