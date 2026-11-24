"""Ready-to-use ZenML dataset steps."""

import json
import os
from typing import Any

import pandas as pd
from zenml import step

from .edc import retrieve
from .materializers import DatasetFileMaterializer
from .runtime import secret_values, transfer_config
from .storage import staging_connection
from .types import DatasetFile, SDKError


def fetch_dataset(dataset_id: str, asset_id: str | None) -> DatasetFile:
    config = transfer_config()
    values = secret_values("dataspace", "EDC_API_KEY")
    headers = {os.environ.get("EDC_API_KEY_HEADER") or "X-Api-Key": values["EDC_API_KEY"]}
    return retrieve(dataset_id, asset_id, config, headers, staging_connection(push=config.mode == "push"))


def read_delimited(dataset: DatasetFile, delimiter: str) -> pd.DataFrame:
    with dataset.open("rb") as stream:
        return pd.read_csv(stream, sep=delimiter)


def read_parquet(dataset: DatasetFile) -> pd.DataFrame:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise SDKError("Reading Parquet needs pyarrow. Install 6g-dali-mlops[parquet].") from None
    with dataset.open("rb") as stream:
        return pd.read_parquet(stream)


def read_json(dataset: DatasetFile) -> Any:
    with dataset.open("rb") as stream:
        try:
            return json.loads(stream.read().decode("utf-8"))
        except ValueError:
            raise SDKError("The distribution is not valid UTF-8 JSON.") from None


def read_jsonl(dataset: DatasetFile) -> pd.DataFrame:
    with dataset.open("rb") as stream:
        try:
            return pd.read_json(stream, lines=True)
        except ValueError:
            raise SDKError("The distribution is not valid JSON Lines.") from None


@step(enable_cache=False, output_materializers=DatasetFileMaterializer)
def retrieve_dataset(dataset_id: str, asset_id: str | None = None) -> DatasetFile:
    return fetch_dataset(dataset_id, asset_id)


@step(enable_cache=False)
def retrieve_csv(dataset_id: str, asset_id: str | None = None) -> pd.DataFrame:
    return read_delimited(fetch_dataset(dataset_id, asset_id), ",")


@step(enable_cache=False)
def retrieve_tsv(dataset_id: str, asset_id: str | None = None) -> pd.DataFrame:
    return read_delimited(fetch_dataset(dataset_id, asset_id), "\t")


@step(enable_cache=False)
def retrieve_parquet(dataset_id: str, asset_id: str | None = None) -> pd.DataFrame:
    return read_parquet(fetch_dataset(dataset_id, asset_id))


@step(enable_cache=False)
def retrieve_json(dataset_id: str, asset_id: str | None = None) -> Any:
    return read_json(fetch_dataset(dataset_id, asset_id))


@step(enable_cache=False)
def retrieve_jsonl(dataset_id: str, asset_id: str | None = None) -> pd.DataFrame:
    return read_jsonl(fetch_dataset(dataset_id, asset_id))
