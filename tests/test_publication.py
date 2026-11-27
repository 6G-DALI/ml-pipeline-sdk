from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO

import boto3
from botocore.exceptions import ClientError
from moto import mock_aws
import pytest

from dali_mlops import Checkpoint, ExportedModel, SDKError
from dali_mlops.metrics import normalize_metrics
from dali_mlops.publication import BUCKET, load_base, publish

VERSION = "2fe3980d-eac4-4f38-a0a6-8d4baa44c508"
MODEL = ExportedModel.from_bytes(b"model data", "model.bin", "application/octet-stream")


@mock_aws
def test_publish_concurrent_retries_and_conflict():
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)

    def run(model):
        return publish(s3, model, {"rmse": 1, "custom": 2}, VERSION, "test", "example", [{"mae": 3}])

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, [MODEL, MODEL]))
    assert results[0] == results[1]
    result = results[0]
    assert result["run_name"] == "test"
    assert result["experiment_name"] == "example"
    assert result["status"] == "SUCCESS"
    assert result["metrics"] == {"model-rmse": 1.0, "custom": 2.0}
    assert result["metrics_history"] == [{"model-mae": 3.0}]
    assert result["model"] == {
        "dataspace_model_id": VERSION, "filename": "model.bin",
        "datalake_uri": f"s3://{BUCKET}/{VERSION}/model.bin", "datalake_size_bytes": len(MODEL.data),
        "media_type": MODEL.media_type, "sha256": sha256(MODEL.data).hexdigest(),
    }
    with pytest.raises(SDKError, match="conflict"):
        run(ExportedModel.from_bytes(b"other data", MODEL.filename, MODEL.media_type))
    assert [obj["Key"] for obj in s3.list_objects_v2(Bucket=BUCKET)["Contents"]] == [f"{VERSION}/model.bin"]
    assert s3.get_object(Bucket=BUCKET, Key=f"{VERSION}/model.bin")["Body"].read() == MODEL.data


@pytest.mark.parametrize("metrics", [{"rmse": 1, "model-rmse": 2}, {"x": float("nan")}, {"x": True}])
def test_invalid_metrics(metrics):
    with pytest.raises(ValueError):
        normalize_metrics(metrics)


def test_failed_or_partial_upload():
    class FakeS3:
        def put_object(self, **kwargs):
            assert kwargs["IfNoneMatch"] == "*"
            if fail:
                raise ClientError({"Error": {"Message": "secret URL", "Code": "InternalError"}}, "PutObject")

        def get_object(self, **kwargs):
            return {"Body": BytesIO(MODEL.data[:3]), "ContentLength": len(MODEL.data)}

    fail = True
    with pytest.raises(SDKError, match=r"^Could not conditionally upload the model\..*support\.$"):
        publish(FakeS3(), MODEL, {}, VERSION, "test")
    fail = False
    with pytest.raises(SDKError, match="failed Data Lake byte verification"):
        publish(FakeS3(), MODEL, {}, VERSION, "test")


CHECKPOINT = Checkpoint.from_bytes(b"checkpoint data", "checkpoint.joblib")
OTHER_VERSION = "a1111111-2222-4333-8444-55555555555b"
BASE_URI = f"s3://{BUCKET}/{OTHER_VERSION}/checkpoint.joblib"


@pytest.fixture
def lake():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        yield s3


def put_base(s3, data=b"base bytes", key=f"{OTHER_VERSION}/checkpoint.joblib", digest=None, **kwargs):
    metadata = {} if digest is False else {"sha256": digest or sha256(data).hexdigest()}
    s3.put_object(Bucket=BUCKET, Key=key, Body=data, Metadata=metadata, ContentType="application/octet-stream", **kwargs)


def test_publish_without_resumable_has_the_old_shape(lake):
    result = publish(lake, MODEL, {"rmse": 1}, VERSION, "test", checkpoint=CHECKPOINT, base_model_uri=BASE_URI)
    assert set(result) == {"run_name", "experiment_name", "status", "metrics", "model"}
    assert [o["Key"] for o in lake.list_objects_v2(Bucket=BUCKET)["Contents"]] == [f"{VERSION}/model.bin"]


@pytest.mark.parametrize("base_uri", [None, BASE_URI])
def test_publish_with_checkpoint(lake, base_uri):
    result = publish(
        lake, MODEL, {"rmse": 1}, VERSION, "test", checkpoint=CHECKPOINT, base_model_uri=base_uri, resumable=True,
    )
    assert result["checkpoint"] == {
        "dataspace_model_id": VERSION, "filename": "checkpoint.joblib",
        "datalake_uri": f"s3://{BUCKET}/{VERSION}/checkpoint.joblib", "datalake_size_bytes": len(CHECKPOINT.data),
        "media_type": "application/octet-stream", "sha256": sha256(CHECKPOINT.data).hexdigest(),
    }
    assert result["base_model_uri"] == base_uri
    assert set(result["model"]) == set(result["checkpoint"])
    stored = lake.get_object(Bucket=BUCKET, Key=f"{VERSION}/checkpoint.joblib")
    assert stored["Body"].read() == CHECKPOINT.data and stored["Metadata"]["sha256"] == result["checkpoint"]["sha256"]
    # an identical retry succeeds, different checkpoint bytes conflict
    assert publish(lake, MODEL, {}, VERSION, "test", checkpoint=CHECKPOINT, resumable=True)["checkpoint"] == result["checkpoint"]
    with pytest.raises(SDKError, match="Checkpoint publication conflict"):
        publish(lake, MODEL, {}, VERSION, "test", checkpoint=Checkpoint.from_bytes(b"other", "checkpoint.joblib"), resumable=True)


def test_equal_stems_are_rejected_before_any_upload(lake):
    for name in ("model.joblib", "model.pt"):
        with pytest.raises(ValueError, match="different file names"):
            publish(lake, MODEL, {}, VERSION, "test", checkpoint=Checkpoint.from_bytes(b"x", name), resumable=True)
    assert "Contents" not in lake.list_objects_v2(Bucket=BUCKET)


def test_resumable_needs_a_checkpoint(lake):
    with pytest.raises(TypeError, match="Checkpoint"):
        publish(lake, MODEL, {}, VERSION, "test", resumable=True)


def test_the_checkpoint_is_published_before_the_model():
    calls = []

    class Recorder:
        def put_object(self, **kwargs):
            calls.append(kwargs["Key"])
            self.last = kwargs

        def get_object(self, **kwargs):
            body = self.last["Body"]
            return {"Body": BytesIO(body), "ContentLength": len(body)}

    publish(Recorder(), MODEL, {}, VERSION, "test", checkpoint=CHECKPOINT, resumable=True)
    assert calls == [f"{VERSION}/checkpoint.joblib", f"{VERSION}/model.bin"]


def test_a_failed_checkpoint_leaves_no_model(lake):
    class Failing:
        def __getattr__(self, name):
            return getattr(lake, name)

        def put_object(self, **kwargs):
            if kwargs["Key"].endswith("checkpoint.joblib"):
                raise ClientError({"Error": {"Code": "InternalError"}}, "PutObject")
            return lake.put_object(**kwargs)

    with pytest.raises(SDKError, match="upload the checkpoint"):
        publish(Failing(), MODEL, {}, VERSION, "test", checkpoint=CHECKPOINT, resumable=True)
    assert "Contents" not in lake.list_objects_v2(Bucket=BUCKET)


def test_load_base(lake):
    put_base(lake)
    base = load_base(BASE_URI, lake)
    assert (base.filename, base.media_type, base.data) == ("checkpoint.joblib", "application/octet-stream", b"base bytes")


@pytest.mark.parametrize("uri", [
    None, "", "https://x/y", f"s3://other-bucket/{OTHER_VERSION}/checkpoint.joblib", f"s3://{BUCKET}/checkpoint.joblib",
    f"s3://{BUCKET}/not-a-uuid/checkpoint.joblib", f"s3://{BUCKET}/{OTHER_VERSION}/a/b.joblib",
    f"s3://{BUCKET}/{OTHER_VERSION}/..", f"{BASE_URI}?x=1", f"s3://{BUCKET}/{OTHER_VERSION.upper()}/checkpoint.joblib",
])
def test_load_base_rejects_wrong_uris(lake, uri):
    with pytest.raises(SDKError, match="must look like"):
        load_base(uri, lake)


def test_load_base_missing_object(lake):
    with pytest.raises(SDKError, match="Could not read the base checkpoint"):
        load_base(BASE_URI, lake)


def test_load_base_sha_mismatch_or_missing(lake):
    put_base(lake, digest="0" * 64)
    with pytest.raises(SDKError, match="does not match the sha256"):
        load_base(BASE_URI, lake)
    put_base(lake, digest=False)
    with pytest.raises(SDKError, match="no sha256"):
        load_base(BASE_URI, lake)


def test_load_base_truncated_or_empty(lake):
    class Truncated:
        def get_object(self, **kwargs):
            return {"Body": BytesIO(b"abc"), "ContentLength": 10, "Metadata": {"sha256": sha256(b"abc").hexdigest()}}

    with pytest.raises(SDKError, match="ended before"):
        load_base(BASE_URI, Truncated())
    put_base(lake, data=b"")
    with pytest.raises(SDKError, match="empty"):
        load_base(BASE_URI, lake)


def test_load_base_roundtrips_a_published_checkpoint(lake):
    result = publish(lake, MODEL, {}, VERSION, "test", checkpoint=CHECKPOINT, resumable=True)
    assert load_base(result["checkpoint"]["datalake_uri"], lake) == CHECKPOINT
