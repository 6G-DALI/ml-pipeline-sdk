from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit

from io import BytesIO

import boto3
import joblib
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
from examples.epoch_mlp import epoch_mlp_pipeline
from examples.random_forest import FEATURE_NAMES, preprocess, random_forest_pipeline

FIXTURE = Path(__file__).parents[1] / "examples" / "telemetry.csv"
VERSION = "96e968f1-f414-40d0-b4dd-8a37897edf74"
PAYLOAD = {}


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
    PAYLOAD.clear()
    PAYLOAD["data"] = FIXTURE.read_bytes()
    with mock_aws():
        staging_s3 = boto3.client("s3", region_name="us-east-1")
        staging_s3.create_bucket(Bucket="datasets")
        datalake_s3 = boto3.client("s3", region_name="us-west-2")
        datalake_s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
        staging = storage.Staging(staging_s3, "datasets")

        def staged_fixture(dataset_id, asset_id):
            key = staging.new_key("telemetry.csv")
            staging_s3.put_object(Bucket="datasets", Key=key, Body=PAYLOAD["data"])
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
    assert set(result["metrics"]) == {"model-mae", "model-rmse", "model-r2"}
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
    assert sorted(obj["Key"] for obj in offline_zenml.list_objects_v2(Bucket=BUCKET)["Contents"]) == [
        f"{VERSION}/checkpoint.joblib", key,
    ]
    assert result["base_model_uri"] is None
    assert result["checkpoint"]["datalake_uri"] == f"s3://{BUCKET}/{VERSION}/checkpoint.joblib"
    forest = joblib.load(BytesIO(offline_zenml.get_object(Bucket=BUCKET, Key=f"{VERSION}/checkpoint.joblib")["Body"].read()))
    assert forest.n_estimators == 50 and len(forest.estimators_) == 50


def test_epoch_example_reports_history(offline_zenml):
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    table = pd.DataFrame({"a": rng.random(40), "b": rng.random(40)})
    table["target"] = table["a"] * 2 + table["b"]
    PAYLOAD["data"] = table.to_csv(index=False).encode()
    run = epoch_mlp_pipeline(dataset_id="numbers", dataspace_model_id=VERSION, run_name="local-epochs", epochs=3)
    result = run.steps["save_model"].output.load()
    assert set(result["metrics"]) == {"loss", "model-mae"}
    assert [point["epoch"] for point in result["metrics_history"]] == [0, 1, 2]
    assert set(result["metrics_history"][0]) == {"epoch", "loss", "model-mae"}
    assert result["model"]["datalake_uri"] == f"s3://{BUCKET}/{VERSION}/model.onnx"
    assert result["checkpoint"]["filename"] == "checkpoint.pt" and result["base_model_uri"] is None

    second_id = "0b6a2f7e-5c1d-4e8a-9f3b-7d2e4a6c8b10"
    resumed = epoch_mlp_pipeline(
        dataset_id="numbers", dataspace_model_id=second_id, run_name="local-epochs-2", epochs=2,
        base_model_uri=result["checkpoint"]["datalake_uri"],
    ).steps["save_model"].output.load()
    assert [point["epoch"] for point in resumed["metrics_history"]] == [3, 4]
    assert resumed["base_model_uri"] == result["checkpoint"]["datalake_uri"]
    assert resumed["metrics"]["loss"] < result["metrics_history"][0]["loss"]

    PAYLOAD["data"] = table.assign(extra=1.0).to_csv(index=False).encode()
    with pytest.raises(Exception):
        epoch_mlp_pipeline(
            dataset_id="numbers", dataspace_model_id="1c7b3a8f-6d2e-4f9b-8a4c-8e3f5b7d9c21", run_name="x",
            base_model_uri=result["checkpoint"]["datalake_uri"],
        )


def test_random_forest_resumes_from_a_base_checkpoint(offline_zenml):
    first = random_forest_pipeline(
        dataset_id="telemetry", dataspace_model_id=VERSION, run_name="v1", n_estimators=10, extra_trees=5,
    ).steps["save_model"].output.load()
    assert first["base_model_uri"] is None
    second_id = "0b6a2f7e-5c1d-4e8a-9f3b-7d2e4a6c8b10"
    run = random_forest_pipeline(
        dataset_id="telemetry", dataspace_model_id=second_id, run_name="v2", n_estimators=10, extra_trees=5,
        base_model_uri=first["checkpoint"]["datalake_uri"],
    )
    result = run.steps["save_model"].output.load()
    assert result["base_model_uri"] == first["checkpoint"]["datalake_uri"]
    assert result["checkpoint"]["datalake_uri"] == f"s3://{BUCKET}/{second_id}/checkpoint.joblib"
    assert result["model"]["datalake_uri"] == f"s3://{BUCKET}/{second_id}/model.onnx"
    forest = joblib.load(BytesIO(offline_zenml.get_object(Bucket=BUCKET, Key=f"{second_id}/checkpoint.joblib")["Body"].read()))
    assert len(forest.estimators_) == 15
    base_forest = joblib.load(BytesIO(offline_zenml.get_object(Bucket=BUCKET, Key=f"{VERSION}/checkpoint.joblib")["Body"].read()))
    assert all(a is not b for a, b in zip(forest.estimators_[:10], base_forest.estimators_))
    # the base trees are kept, the new ones are fitted on the new data
    assert forest.estimators_[0].tree_.node_count == base_forest.estimators_[0].tree_.node_count


def test_random_forest_rejects_a_base_with_other_features(offline_zenml):
    from sklearn.ensemble import RandomForestRegressor

    from dali_mlops import dump_joblib

    wrong = RandomForestRegressor(n_estimators=2).fit(np.random.rand(10, 2), np.random.rand(10))
    checkpoint = dump_joblib(wrong, "old.joblib")
    key = "a1111111-2222-4333-8444-55555555555b/old.joblib"
    offline_zenml.put_object(
        Bucket=BUCKET, Key=key, Body=checkpoint.data, Metadata={"sha256": sha256(checkpoint.data).hexdigest()},
    )
    with pytest.raises(Exception):
        random_forest_pipeline(
            dataset_id="telemetry", dataspace_model_id=VERSION, run_name="v2", base_model_uri=f"s3://{BUCKET}/{key}",
        )
    assert "Contents" not in offline_zenml.list_objects_v2(Bucket=BUCKET, Prefix=VERSION)


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
            "retrieve_csv", "load_base_checkpoint", "preprocess", "train", "evaluate", "export", "save_model",
        }
        parameters = snapshot.pipeline_configuration.parameters
        assert parameters["dataset_id"] == "telemetry" and parameters["n_estimators"] == 50
    finally:
        Client._reset_instance()
