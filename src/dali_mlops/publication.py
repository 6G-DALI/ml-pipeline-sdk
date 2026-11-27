"""Conditionally publish already-exported model bytes and report the verified result."""

from contextlib import closing
from hashlib import sha256
import json
import logging
from pathlib import PurePath
from urllib.parse import urlsplit
from uuid import UUID

from botocore.exceptions import ClientError
from zenml import log_metadata, step

from .config import DEFAULTS, datalake_bucket
from .materializers import CheckpointMaterializer
from .metrics import normalize_history, normalize_metrics
from .runtime import secret_values
from .storage import s3_client, storage_errors
from .types import Checkpoint, ExportedModel, SDKError, safe_filename

BUCKET = DEFAULTS["DLAKE_BUCKET"]
logger = logging.getLogger(__name__)
LARGE_CHECKPOINT_BYTES = 500 * 1024**2


def model_version(value):
    try:
        return str(UUID(value))
    except (ValueError, AttributeError, TypeError):
        raise ValueError("dataspace_model_id must be a trained model version UUID.") from None


def verify_model(client, bucket, key, size, digest, media_type, *, existing=False, label="model"):
    with storage_errors(f"Could not read back the {label} from the Data Lake. Check GET permissions and storage health."):
        result = client.get_object(Bucket=bucket, Key=key)
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
        message = (f"{label.capitalize()} publication conflict. Different bytes already exist at this version and filename."
                   if existing else f"The uploaded {label} failed Data Lake byte verification.")
        raise SDKError(message)
    return result.get("ContentType") or media_type


def upload_model(client, model, version, bucket=BUCKET, label="model"):
    key = f"{version}/{model.filename}"
    size = len(model.data)
    digest = sha256(model.data).hexdigest()
    existing = False
    with storage_errors(
        f"Could not conditionally upload the {label}. Check the Data Lake connection and If-None-Match support."
    ):
        try:
            client.put_object(
                Bucket=bucket, Key=key, Body=model.data, ContentType=model.media_type,
                Metadata={"sha256": digest}, IfNoneMatch="*",
            )
        except ClientError as error:
            status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status == 409:
                raise SDKError("The Data Lake reported a competing write. Retry publication for this version.") from None
            if status != 412:
                raise
            existing = True
    media_type = verify_model(client, bucket, key, size, digest, model.media_type, existing=existing, label=label)
    return {
        "dataspace_model_id": version, "filename": model.filename,
        "datalake_uri": f"s3://{bucket}/{key}", "datalake_size_bytes": size,
        "media_type": media_type, "sha256": digest,
    }


def stem(filename):
    return PurePath(filename).stem


def publish(
    client, model, metrics, dataspace_model_id, run_name, experiment_name="", metric_history=None, bucket=BUCKET,
    checkpoint=None, base_model_uri=None, resumable=False,
):
    version = model_version(dataspace_model_id)
    final_metrics = normalize_metrics(metrics)
    history = normalize_history(metric_history) if metric_history is not None else None
    if resumable:
        if not isinstance(checkpoint, Checkpoint):
            raise TypeError("A resumable pipeline must return a Checkpoint.")
        if stem(checkpoint.filename) == stem(model.filename):
            raise ValueError(
                "The checkpoint and the model need different file names without the extension "
                f"('{checkpoint.filename}' and '{model.filename}'): the platform derives asset ids from the stem."
            )
        if len(checkpoint.data) > LARGE_CHECKPOINT_BYTES:
            logger.warning("The checkpoint is %d MiB. Large checkpoints slow down later runs.", len(checkpoint.data) >> 20)
    result = {
        "run_name": run_name, "experiment_name": experiment_name or None,
        "status": "SUCCESS", "metrics": final_metrics,
    }
    # The checkpoint goes first so that a failed checkpoint never leaves a model without its checkpoint.
    uploaded_checkpoint = upload_model(client, checkpoint, version, bucket, "checkpoint") if resumable else None
    result["model"] = upload_model(client, model, version, bucket)
    if history is not None:
        result["metrics_history"] = history
    if resumable:
        result["checkpoint"] = uploaded_checkpoint
        result["base_model_uri"] = base_model_uri
    return result


def parse_base_uri(uri, bucket):
    """``s3://<bucket>/<uuid>/<file>`` in the Data Lake bucket, nothing else."""
    parsed = urlsplit(uri) if isinstance(uri, str) else None
    parts = parsed.path.lstrip("/").split("/") if parsed else []
    try:
        if not parsed or parsed.scheme != "s3" or parsed.netloc != bucket or parsed.query or parsed.fragment:
            raise ValueError
        if len(parts) != 2 or str(UUID(parts[0])) != parts[0]:
            raise ValueError
        safe_filename(parts[1])
    except ValueError:
        raise SDKError(f"The base model URI must look like s3://{bucket}/<version uuid>/<checkpoint file>.") from None
    return parts[0], parts[1]


def load_base(uri, client=None, bucket=None, media_type="application/octet-stream"):
    """Download the base version's checkpoint from the Data Lake and verify size and SHA-256."""
    bucket = bucket or datalake_bucket()
    version, filename = parse_base_uri(uri, bucket)
    key = f"{version}/{filename}"
    client = client or datalake_client()
    with storage_errors("Could not read the base checkpoint from the Data Lake. Check that it exists and the GET permission."):
        result = client.get_object(Bucket=bucket, Key=key)
        with closing(result["Body"]) as stream:
            data = stream.read()
    digest = {k.lower(): v for k, v in (result.get("Metadata") or {}).items()}.get("sha256")
    if not data:
        raise SDKError("The base checkpoint in the Data Lake is empty.")
    if result.get("ContentLength", len(data)) != len(data):
        raise SDKError("The base checkpoint download ended before its expected size.")
    if not digest:
        raise SDKError("The base checkpoint has no sha256 recorded in the Data Lake, so it cannot be verified.")
    if sha256(data).hexdigest() != digest:
        raise SDKError("The base checkpoint does not match the sha256 recorded in the Data Lake.")
    return Checkpoint(filename, result.get("ContentType") or media_type, data)


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
    metric_history: list[dict[str, float | int]] | None = None,
    checkpoint: Checkpoint | None = None,
    base_model_uri: str | None = None,
    resumable: bool = False,
) -> dict:
    """Publish to the central Data Lake and report the MO result."""
    result = publish(
        datalake_client(), model, metrics, dataspace_model_id, run_name, experiment_name, metric_history,
        datalake_bucket(), checkpoint, base_model_uri, resumable,
    )
    log_metadata(metadata={"api_result": result, "metrics": result["metrics"]})
    logger.info("api_result=%s", json.dumps(result, allow_nan=False))
    return result


@step(enable_cache=False, output_materializers=CheckpointMaterializer)
def load_base_checkpoint(base_model_uri: str | None = None) -> Checkpoint | None:
    """The base version's checkpoint, or None for a from-scratch run."""
    return load_base(base_model_uri) if base_model_uri else None
