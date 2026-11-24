from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO

import boto3
from botocore.exceptions import ClientError
from moto import mock_aws
import pytest

from dali_mlops import ExportedModel, SDKError
from dali_mlops.metrics import normalize_metrics
from dali_mlops.publication import BUCKET, publish

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
