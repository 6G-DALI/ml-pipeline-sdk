"""Ready-to-use ZenML dataset steps."""

import os

import pandas as pd
from zenml import step

from .edc import retrieve
from .materializers import DatasetFileMaterializer
from .runtime import secret_values, transfer_config
from .storage import staging_connection
from .types import DatasetFile


def fetch_dataset(dataset_id: str, asset_id: str | None) -> DatasetFile:
    config = transfer_config()
    values = secret_values("dataspace", "EDC_API_KEY")
    headers = {os.environ.get("EDC_API_KEY_HEADER") or "X-Api-Key": values["EDC_API_KEY"]}
    return retrieve(dataset_id, asset_id, config, headers, staging_connection(push=config.mode == "push"))


def read_delimited(dataset: DatasetFile, delimiter: str) -> pd.DataFrame:
    with dataset.open("rb") as stream:
        return pd.read_csv(stream, sep=delimiter)


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
    raise NotImplementedError('Use retrieve_dataset() and DatasetFile.open("rb") to parse Parquet.')


@step(enable_cache=False)
def retrieve_json(dataset_id: str, asset_id: str | None = None) -> object:
    raise NotImplementedError('Use retrieve_dataset() and DatasetFile.open("rb") to decode JSON.')


@step(enable_cache=False)
def retrieve_jsonl(dataset_id: str, asset_id: str | None = None) -> pd.DataFrame:
    raise NotImplementedError('Use retrieve_dataset() and DatasetFile.open("rb") to parse JSON Lines.')
