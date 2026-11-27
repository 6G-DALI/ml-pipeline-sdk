"""Python steps and artifact types for the 6G-DALI MLOps SDK.

Names are loaded on first use so that ``import dali_mlops.spec`` does not import ZenML.
"""

from importlib import import_module

__version__ = "0.3.0"

_EXPORTS = {
    "Checkpoint": "types", "DatasetFile": "types", "ExportedModel": "types", "SDKError": "types", "TrainingOutput": "types",
    "retrieve_csv": "retrieval", "retrieve_dataset": "retrieval", "retrieve_json": "retrieval",
    "retrieve_jsonl": "retrieval", "retrieve_parquet": "retrieval", "retrieve_tsv": "retrieval",
    "save_model": "publication",
    "dump_joblib": "checkpoint", "load_joblib": "checkpoint", "dump_torch_state": "checkpoint",
    "load_torch_state": "checkpoint", "dali_pipeline": "pipeline",
}

__all__ = sorted(_EXPORTS) + ["__version__"]


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
