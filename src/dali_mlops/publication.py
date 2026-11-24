"""Conditionally publish already-exported model bytes and report the verified result."""

from contextlib import closing
from hashlib import sha256
import json
import logging
from uuid import UUID

from botocore.exceptions import ClientError
from zenml import log_metadata, step

from .metrics import normalize_history, normalize_metrics
from .runtime import secret_values
from .storage import s3_client, storage_errors
from .types import ExportedModel, SDKError

BUCKET = "6g-dali-models"
logger = logging.getLogger(__name__)


def model_version(value):
    try:
        return str(UUID(value))
    except (ValueError, AttributeError, TypeError):
        raise ValueError("dataspace_model_id must be a trained model version UUID.") from None


def verify_model(client, key, size, digest, media_type, *, existing=False):
    with storage_errors("Could not read back the model from the Data Lake. Check GET permissions and storage health."):
        result = client.get_object(Bucket=BUCKET, Key=key)
        with closing(result["Body"]) as stream:
            if result["ContentLength"] != size:
                matches = False
            else:
                actual = sha256()
                count = 0
                while chunk := stream.read(1024 * 1024):
                    count += len(chunk)
                    actual.update(chunk)
                matches = count == size and actual.hexdigest() == digest
    if not matches:
        message = ("Model publication conflict. Different bytes already exist at this version and filename."
                   if existing else "The uploaded model failed Data Lake byte verification.")
        raise SDKError(message)
    return result.get("ContentType") or media_type


def upload_model(client, model, version):
    key = f"{version}/{model.filename}"
    size = len(model.data)
    digest = sha256(model.data).hexdigest()
    existing = False
    with storage_errors("Could not conditionally upload the model. Check the Data Lake connection and If-None-Match support."):
        try:
            client.put_object(
                Bucket=BUCKET, Key=key, Body=model.data, ContentType=model.media_type,
                Metadata={"sha256": digest}, IfNoneMatch="*",
            )
        except ClientError as error:
            status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status == 409:
                raise SDKError("The Data Lake reported a competing write. Retry publication for this version.") from None
            if status != 412:
                raise
            existing = True
    media_type = verify_model(client, key, size, digest, model.media_type, existing=existing)
    return {
        "dataspace_model_id": version, "filename": model.filename,
        "datalake_uri": f"s3://{BUCKET}/{key}", "datalake_size_bytes": size,
        "media_type": media_type, "sha256": digest,
    }


def publish(client, model, metrics, dataspace_model_id, run_name, experiment_name="", metric_history=None):
    version = model_version(dataspace_model_id)
    final_metrics = normalize_metrics(metrics)
    history = normalize_history(metric_history) if metric_history is not None else None
    result = {
        "run_name": run_name, "experiment_name": experiment_name or None,
        "status": "SUCCESS", "metrics": final_metrics,
        "model": upload_model(client, model, version),
    }
    if history is not None:
        result["metrics_history"] = history
    return result


def datalake_client():
    secret = secret_values("datalake-model-storage", "DLAKE_ENDPOINT", "DLAKE_ACCESS_KEY", "DLAKE_SECRET_KEY")
    with storage_errors("Could not initialize the Data Lake S3 client."):
        return s3_client(
            endpoint_url=secret["DLAKE_ENDPOINT"], aws_access_key_id=secret["DLAKE_ACCESS_KEY"],
            aws_secret_access_key=secret["DLAKE_SECRET_KEY"], region_name=secret.get("DLAKE_REGION") or "us-east-1",
        )


@step(enable_cache=False)
def save_model(
    model: ExportedModel,
    metrics: dict[str, float],
    dataspace_model_id: str,
    run_name: str,
    experiment_name: str = "",
    metric_history: list[dict[str, float]] | None = None,
) -> dict:
    """Publish to the central Data Lake and report the MO result."""
    result = publish(datalake_client(), model, metrics, dataspace_model_id, run_name, experiment_name, metric_history)
    log_metadata(metadata={"api_result": result, "metrics": result["metrics"]})
    logger.info("api_result=%s", json.dumps(result, allow_nan=False))
    return result
