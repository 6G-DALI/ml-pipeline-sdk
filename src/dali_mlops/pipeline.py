"""The ``@dali_pipeline`` decorator: the author writes only the ML logic."""

import functools
import inspect
from pathlib import Path
import typing

from zenml import pipeline as zenml_pipeline
from zenml.config import DockerSettings

from .publication import load_base_checkpoint, save_model
from .retrieval import (
    retrieve_csv, retrieve_dataset, retrieve_json, retrieve_jsonl, retrieve_parquet, retrieve_tsv,
)
from .spec import DATASET_FORMATS, RESERVED, RESERVED_NAMES
from .types import Checkpoint, TrainingOutput

RETRIEVAL = {
    "csv": retrieve_csv, "tsv": retrieve_tsv, "parquet": retrieve_parquet, "json": retrieve_json,
    "jsonl": retrieve_jsonl, "raw": retrieve_dataset,
}
RESERVED_ANNOTATIONS = {
    "dataset_id": str, "dataspace_model_id": str, "run_name": str, "asset_id": str | None, "experiment_name": str,
    "base_model_uri": str | None,
}
KEYWORD = inspect.Parameter.KEYWORD_ONLY


def docker_settings(base_image, requirements):
    """ZenML builds an image when requirements are given; a bare base image is used as is."""
    if isinstance(requirements, Path):
        requirements = str(requirements)
    if not base_image and not requirements:
        return None
    return DockerSettings(parent_image=base_image, requirements=requirements or None, skip_build=not requirements)


def unpack(result, resumable=False):
    """Return ``(model, metrics, history)``, plus the checkpoint for a resumable pipeline, which must provide it."""
    if isinstance(result, TrainingOutput):
        parts = (result.model, result.metrics, result.metric_history, result.checkpoint)
    elif isinstance(result, tuple) and len(result) in (2, 3) + ((4,) if resumable else ()):
        parts = (*result, *([None] * (4 - len(result))))
    elif resumable:
        raise TypeError(
            "A resumable @dali_pipeline body must return TrainingOutput(model, metrics, metric_history=None, "
            "checkpoint=...) or a (model, metrics, metric_history, checkpoint) tuple."
        )
    else:
        raise TypeError(
            "A @dali_pipeline body must return TrainingOutput(model, metrics, metric_history=None) "
            "or a (model, metrics) tuple."
        )
    if resumable and parts[3] is None:
        raise TypeError("A resumable @dali_pipeline body must return a checkpoint (TrainingOutput.checkpoint).")
    return parts if resumable else parts[:3]


def signature_of(function, has_dataset, resumable=False):
    """The pipeline signature: the reserved parameters of the run contract, then the author's own."""
    hints = typing.get_type_hints(function)
    own = list(inspect.signature(function).parameters.values())
    if has_dataset:
        if not own or own[0].kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            raise TypeError(f"{function.__name__} must take the loaded dataset as its first parameter.")
        own = own[1:]
    if resumable:
        base = own[0] if own else None
        if base is None or base.name != "base" or base.kind not in (
            inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise TypeError(
                f"{function.__name__} is resumable and must take the parameter 'base' right after the dataset."
            )
        if base.default is not None:
            raise TypeError(f"The parameter 'base' of {function.__name__} must default to None.")
        if hints.get("base") not in (Checkpoint | None, typing.Optional[Checkpoint]):
            raise TypeError(f"The parameter 'base' of {function.__name__} must be annotated Checkpoint | None.")
        own = own[1:]
    for parameter in own:
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError(f"{function.__name__} cannot use *args or **kwargs.")
    clash = sorted(RESERVED_NAMES.intersection(parameter.name for parameter in own))
    if clash:
        raise TypeError(f"{function.__name__} declares reserved parameters {clash}. The platform sets them.")
    own = [parameter.replace(kind=KEYWORD, annotation=hints.get(parameter.name, parameter.annotation)) for parameter in own]
    reserved = [
        inspect.Parameter(
            item["name"], KEYWORD, annotation=RESERVED_ANNOTATIONS[item["name"]],
            default=inspect.Parameter.empty if item["required"] else item["default"],
        )
        for item in RESERVED if (has_dataset or not item["dataset"]) and (resumable or not item["resumable"])
    ]
    merged = reserved + own
    ordered = [p for p in merged if p.default is p.empty] + [p for p in merged if p.default is not p.empty]
    return inspect.Signature(ordered, return_annotation=dict)


def build(function, *, has_dataset, retrieval, resumable=False):
    signature = signature_of(function, has_dataset, resumable)

    @functools.wraps(function)
    def pipeline_body(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        reserved = {name: values.pop(name) for name in RESERVED_NAMES if name in values}
        leading = []
        if has_dataset:
            leading.append(retrieval(reserved["dataset_id"], reserved["asset_id"]))
        if resumable:
            leading.append(load_base_checkpoint(reserved["base_model_uri"]))
        result = function(*leading, **values)
        unpacked = unpack(result, resumable)
        model, metrics, history, checkpoint = unpacked if resumable else (*unpacked, None)
        extra = {"checkpoint": checkpoint, "base_model_uri": reserved["base_model_uri"], "resumable": True} if resumable else {}
        return save_model(
            model, metrics, reserved["dataspace_model_id"], reserved["run_name"], reserved["experiment_name"], history,
            **extra,
        )

    pipeline_body.__signature__ = signature
    pipeline_body.__annotations__ = {
        **{name: parameter.annotation for name, parameter in signature.parameters.items()}, "return": dict,
    }
    return pipeline_body


def dali_pipeline(
    _function=None, *, name=None, dataset="csv", base_image=None, requirements=None, environment=None,
    enable_cache=False, resumable=False, **options,
):
    """Turn a function holding only the ML logic into a ZenML pipeline that honours the run contract.

    ``dataset`` is the format of the input distribution (csv, tsv, parquet, json, jsonl, raw) or None
    for a pipeline without input data. The function receives the loaded dataset as its first argument,
    then its own hyperparameters, and returns ``TrainingOutput(model, metrics, metric_history=None)`` or
    a ``(model, metrics)`` tuple. The decorator adds ``dataset_id``, ``asset_id``, ``dataspace_model_id``,
    ``run_name`` and ``experiment_name`` to the pipeline signature and publishes the model.

    With ``resumable=True`` the function takes ``base: Checkpoint | None = None`` right after the dataset
    (None means from scratch), the decorator adds ``base_model_uri``, downloads and verifies the checkpoint
    and the function must return ``TrainingOutput(..., checkpoint=Checkpoint)``.
    """
    if dataset is not None and dataset not in DATASET_FORMATS:
        raise ValueError(f"dataset must be one of {', '.join(DATASET_FORMATS)} or None.")

    def decorate(function):
        body = build(function, has_dataset=dataset is not None, retrieval=RETRIEVAL.get(dataset), resumable=resumable)
        settings = dict(options.pop("settings", None) or {})
        docker = docker_settings(base_image, requirements)
        if docker is not None:
            settings.setdefault("docker", docker)
        extra = dict(options)
        if settings:
            extra["settings"] = settings
        if environment:
            extra["environment"] = dict(environment)
        return zenml_pipeline(name=name or function.__name__, enable_cache=enable_cache, **extra)(body)

    return decorate(_function) if _function is not None else decorate
