from pathlib import Path
from urllib.parse import urlsplit

import boto3
from moto import mock_aws
import numpy as np
import onnx
import pandas as pd
import pytest
from zenml import pipeline, step
from zenml.client import Client

from dali_mlops import DatasetFile, retrieve_dataset
from dali_mlops import publication, retrieval, storage
from dali_mlops.publication import BUCKET
from examples.random_forest import FEATURE_NAMES, preprocess, random_forest_pipeline

FIXTURE = Path(__file__).parents[1] / "examples" / "telemetry.csv"
VERSION = "96e968f1-f414-40d0-b4dd-8a37897edf74"


@step
def read_staged(dataset: DatasetFile) -> bytes:
    with dataset.open("rb") as stream:
        return stream.read()


@pipeline
def raw_pipeline() -> bytes:
    return read_staged(retrieve_dataset("telemetry"))


@pytest.fixture
def offline_zenml(monkeypatch, tmp_path):
    monkeypatch.setenv("ZENML_CONFIG_PATH", str(tmp_path / "zenml"))
    monkeypatch.setenv("ZENML_ANALYTICS_OPT_IN", "false")
    monkeypatch.setenv("ZENML_ENABLE_REPO_INIT_WARNINGS", "false")
    Client._reset_instance()
    with mock_aws():
        staging_s3 = boto3.client("s3", region_name="us-east-1")
        staging_s3.create_bucket(Bucket="datasets")
        datalake_s3 = boto3.client("s3", region_name="us-west-2")
        datalake_s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
        staging = storage.Staging(staging_s3, "datasets")

        def staged_fixture(dataset_id, asset_id):
            key = staging.new_key("telemetry.csv")
            staging_s3.put_object(Bucket="datasets", Key=key, Body=FIXTURE.read_bytes())
            return DatasetFile(
                "telemetry.csv", "text/csv", dataset_id, asset_id or "fixture", staging.uri(key),
                "agreement", "transfer", staging.verify(key),
            )

        monkeypatch.setattr(retrieval, "fetch_dataset", staged_fixture)
        monkeypatch.setattr(
            storage, "staging_connection",
            lambda *, bucket: storage.Staging(boto3.client("s3", region_name="us-east-1"), bucket),
        )
        monkeypatch.setattr(publication, "datalake_client", lambda: datalake_s3)
        yield datalake_s3
    Client._reset_instance()


def test_raw_pipeline(offline_zenml):
    run = raw_pipeline()
    assert run.steps["read_staged"].output.load() == FIXTURE.read_bytes()
    reference = run.steps["retrieve_dataset"].output.load()
    assert isinstance(reference, DatasetFile)
    assert urlsplit(reference.storage_uri).netloc == "datasets"
    with reference.open("rb") as stream:
        assert stream.read() == FIXTURE.read_bytes()


def test_offline_random_forest(offline_zenml):
    run = random_forest_pipeline(dataset_id="telemetry", dataspace_model_id=VERSION, run_name="local-forest")
    published = run.steps["save_model"]
    result = published.output.load()
    assert published.run_metadata["api_result"] == result
    assert result["status"] == "SUCCESS"
    assert result["run_name"] == "local-forest"
    assert set(result["metrics"]) == {"mae", "model-rmse", "r2"}
    assert "metrics_history" not in result
    exported = run.steps["export"].output.load()
    key = f"{VERSION}/model.onnx"
    body = offline_zenml.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    assert exported.data == body
    graph = onnx.load_model_from_string(body)
    onnx.checker.check_model(graph)
    schema = graph.graph.input[0]
    assert schema.name == "telemetry"
    assert schema.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert schema.type.tensor_type.shape.dim[1].dim_value == len(FEATURE_NAMES)
    assert [obj["Key"] for obj in offline_zenml.list_objects_v2(Bucket=BUCKET)["Contents"]] == [key]


def test_chronological_split_drops_crossing_targets():
    data = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="min", tz="UTC"),
        "container_name": ["api"] * 10,
        "cpu_percent": [f"{i}%" for i in range(10)],
        "memory_usage": ["1GiB / 2GiB"] * 10,
        "pids": [2] * 10,
    })
    train_x, train_y, test_x, test_y = preprocess.entrypoint(data.iloc[::-1], 0.3)
    np.testing.assert_array_equal(train_x[:, 0], np.arange(6))
    np.testing.assert_array_equal(train_x[:, 1], [1024] * 6)
    np.testing.assert_array_equal(train_y, np.arange(1, 7))
    np.testing.assert_array_equal(test_x[:, 0], [7, 8])
    np.testing.assert_array_equal(test_y, [8, 9])


def test_compile_without_external_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("ZENML_CONFIG_PATH", str(tmp_path / "zenml"))
    monkeypatch.setenv("ZENML_ANALYTICS_OPT_IN", "false")
    monkeypatch.delenv("PIVEAU_HUB_URL", raising=False)
    monkeypatch.delenv("EDC_PROVIDER_ID", raising=False)
    Client._reset_instance()
    try:
        random_forest_pipeline.prepare(dataset_id="telemetry", dataspace_model_id=VERSION, run_name="compile-test")
        snapshot, _, _ = random_forest_pipeline._compile()
        assert set(snapshot.step_configurations) == {
            "retrieve_csv", "preprocess", "train", "evaluate", "export", "save_model",
        }
    finally:
        Client._reset_instance()
