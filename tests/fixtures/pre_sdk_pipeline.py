from __future__ import annotations

import logging
import os
from typing import Annotated, Any, ClassVar, Dict, Optional, Tuple, Type

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from zenml import ArtifactConfig, log_metadata, pipeline, step
from zenml.client import Client
from zenml.enums import ArtifactType

from zenml.materializers.base_materializer import BaseMaterializer

from zenml.artifacts.unmaterialized_artifact import UnmaterializedArtifact
from zenml.config import DockerSettings


docker_settings = DockerSettings(
    parent_image="nmgrigor/zenml-kubeflow-runtime:v5",
    skip_build=True,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

FEATURE_NAMES = ["cpu_percent", "memory_used_mib", "pids"]
EDC_POLL_TIMEOUT_SECONDS = 300
EDC_POLL_INTERVAL_SECONDS = 2
DATASET_UPLOAD_URL_EXPIRY_SECONDS = 600


class ONNXArtifact(bytes):
    """ONNX bytes with a dedicated type so ZenML selects ONNXMaterializer."""


class ONNXMaterializer(BaseMaterializer):
    """Save ONNX bytes as model.onnx instead of the default bytes format.

    ZenML has built-in materializers for DataFrames and arrays; this custom
    materializer preserves the ONNX filename using the stack's file interface."""

    ASSOCIATED_TYPES: ClassVar[Tuple[Type[Any], ...]] = (ONNXArtifact,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL
    FILENAME: ClassVar[str] = "model.onnx"

    def save(self, data: ONNXArtifact) -> None:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"Expected ONNX bytes, got {type(data)!r}")

        filepath = os.path.join(self.uri, self.FILENAME)
        with self.artifact_store.open(filepath, "wb") as f:
            f.write(bytes(data))

    def load(self, data_type: Type[Any]) -> ONNXArtifact:
        filepath = os.path.join(self.uri, self.FILENAME)
        with self.artifact_store.open(filepath, "rb") as f:
            return ONNXArtifact(f.read())


def _artifact_file_uri(artifact_uri: str, filename: str) -> str:
    """Build the concrete object URI inside a ZenML artifact URI."""
    return f"{artifact_uri.rstrip('/')}/{filename}"


def _presigned_url(artifact_uri: str, filename: str, expires: int) -> Optional[str]:
    """Return a temporary download URL, or None if presigning is unavailable."""
    artifact_store = Client().active_stack.artifact_store
    object_uri = _artifact_file_uri(artifact_uri, filename)

    filesystem = getattr(artifact_store, "filesystem", None)
    if filesystem is None:
        return None

    url_method = getattr(filesystem, "url", None)
    if not callable(url_method):
        return None

    try:
        return str(url_method(object_uri, expires=expires))
    except Exception:
        logger.exception(
            "The active ZenML artifact store could not generate a presigned URL."
        )
        return None


def _retrieve_dataspace_dataset(
dataset_id: str, asset_id: str,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Negotiate a CSV transfer into the stack's datasets bucket.

    The provider must reach the storage endpoint; it receives a signed PUT
    URL, not storage credentials. Adapted from download_dataset.py."""
    import time
    from urllib.parse import parse_qs, quote, urlsplit

    import requests

    client = Client()
    artifact_store = client.active_stack.artifact_store
    filesystem = getattr(artifact_store, "filesystem", None)
    if not artifact_store.path.startswith("s3://") or not callable(
        getattr(filesystem, "url", None)
    ):
        raise ValueError("Dataset retrieval requires an S3-compatible artifact store with presigning.")
    api_key = client.get_secret("dataspace").secret_values["EDC_API_KEY"]
    hub = os.getenv("PIVEAU_HUB_URL", "https://dataspace.6gdali.eu").rstrip("/")
    provider = os.getenv("EDC_PROVIDER_URL", "https://edc.dataspace.6gdali.eu").rstrip("/")
    consumer = os.getenv("EDC_CONSUMER_URL", provider).rstrip("/")
    provider_id = os.getenv("EDC_PROVIDER_ID", "provider-facility")
    edc = "https://w3id.org/edc/v0.0.1/ns/"
    dsp = f"{provider}/protocol"

    def as_list(value):
        return value if isinstance(value, list) else ([] if value is None else [value])

    def text(value):
        value = as_list(value)
        value = value[0] if value else ""
        if isinstance(value, dict):
            value = value.get("@value") or value.get("@id") or ""
        return str(value)

    # Verify that the selected connector asset belongs to this catalogue dataset.
    response = requests.get(
        f"{hub}/datasets/{quote(dataset_id, safe='')}",
        headers={"Accept": "application/ld+json"}, timeout=30,
    )
    response.raise_for_status()
    distributions = response.json().get("@graph", [])
    if not any(text(node.get("https://dali-project.eu/ns#assetId")) == asset_id
               for node in distributions):
        raise ValueError(f"Asset {asset_id!r} is not a connector distribution of dataset {dataset_id!r}.")

    with requests.Session() as session:
        session.headers.update({os.getenv("EDC_API_KEY_HEADER", "X-Api-Key"): api_key})

        def call(method, path, body=None):
            response = session.request(
                method, f"{consumer}/management/v3{path}", json=body, timeout=30,
            )
            # Do not include response bodies: they may contain the signed URL.
            if not response.ok:
                raise RuntimeError(f"EDC {method} {path} failed (HTTP {response.status_code}).")
            return response.json()

        def wait(path, expected):
            deadline = time.monotonic() + EDC_POLL_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                result = call("GET", path)
                state = result.get("state") or result.get("edc:state")
                logger.info("EDC %s state: %s", path, state)
                if state == expected:
                    return result
                if state in {"TERMINATED", "ERROR"}:
                    raise RuntimeError(f"EDC {path} ended in {state}; inspect connector logs for details.")
                time.sleep(EDC_POLL_INTERVAL_SECONDS)
            raise TimeoutError(
                f"EDC {path} did not reach {expected} within {EDC_POLL_TIMEOUT_SECONDS} seconds."
            )

        catalog = call("POST", "/catalog/request", {
            "@context": {"@vocab": edc}, "counterPartyAddress": dsp,
            "protocol": "dataspace-protocol-http",
            "querySpec": {"filterExpression": [
                {"operandLeft": edc + "id", "operator": "=", "operandRight": asset_id}
            ]},
        })
        entries = as_list(catalog.get("dcat:dataset"))
        offers = as_list(entries[0].get("odrl:hasPolicy")) if entries else []
        if not offers:
            raise ValueError(f"No provider offer available for asset {asset_id!r}.")
        offer = offers[0]
        negotiation = call("POST", "/contractnegotiations", {
            "@context": {"@vocab": edc, "odrl": "http://www.w3.org/ns/odrl/2/"},
            "@type": "ContractRequest", "counterPartyAddress": dsp,
            "providerId": provider_id, "protocol": "dataspace-protocol-http",
            "policy": {
                "@id": offer["@id"], "@type": "http://www.w3.org/ns/odrl/2/Offer",
                "odrl:permission": offer.get("odrl:permission", []),
                "odrl:prohibition": offer.get("odrl:prohibition", []),
                "odrl:obligation": offer.get("odrl:obligation", []),
                "odrl:target": {"@id": asset_id},
                "odrl:assigner": {"@id": provider_id},
            },
        })
        finalized = wait(f"/contractnegotiations/{negotiation['@id']}", "FINALIZED")
        agreement_id = finalized.get("contractAgreementId") or finalized.get("edc:contractAgreementId")
        if not agreement_id:
            raise RuntimeError("Finalized negotiation did not contain a contract agreement ID.")

        # Use the stack's endpoint and credentials with the datasets bucket.
        # Retrieving the same asset again replaces this staged CSV.
        staging_uri = f"s3://datasets/{asset_id}.csv"
        put_url = filesystem.url(
            staging_uri,
            expires=DATASET_UPLOAD_URL_EXPIRY_SECONDS,
            client_method="put_object",
        )

        # Log only the destination, never credentials or signed query values.
        def safe_destination(url):
            parsed = urlsplit(str(url))
            host = parsed.hostname or ""
            if ":" in host:
                host = f"[{host}]"
            port = f":{parsed.port}" if parsed.port else ""
            return f"{parsed.scheme}://{host}{port}{parsed.path}"

        client_options = getattr(filesystem, "client_kwargs", None) or {}
        storage_options = getattr(filesystem, "config_kwargs", None) or {}
        read_endpoint = client_options.get("endpoint_url") or getattr(filesystem, "endpoint_url", None)
        signed_query = parse_qs(urlsplit(put_url).query)
        signing_version = (
            "SigV4" if "X-Amz-Algorithm" in signed_query
            else "SigV2" if "AWSAccessKeyId" in signed_query
            else "unknown"
        )
        logger.info(
            "Dataspace staging: store=%s object=%s read_endpoint=%s",
            artifact_store.name, staging_uri,
            safe_destination(read_endpoint) if read_endpoint else "SDK default",
        )
        logger.info(
            "Dataspace PUT destination=%s signing=%s addressing_style=%s region=%s",
            safe_destination(put_url), signing_version,
            storage_options.get("s3", {}).get("addressing_style", "SDK default"),
            client_options.get("region_name", "SDK default"),
        )
        transfer = call("POST", "/transferprocesses", {
            "@context": {"@vocab": edc}, "@type": "TransferRequest",
            "counterPartyAddress": dsp, "connectorId": provider_id,
            "protocol": "dataspace-protocol-http", "contractId": agreement_id,
            "assetId": asset_id, "transferType": "PresignedHttpData-PUSH",
            "dataDestination": {"type": "PresignedHttpData", "baseUrl": put_url, "method": "PUT"},
        })
        wait(f"/transferprocesses/{transfer['@id']}", "COMPLETED")

    # External uploads can leave cached listings stale, even after COMPLETED.
    filesystem.invalidate_cache(staging_uri)
    filesystem.invalidate_cache("s3://datasets")
    try:
        object_info = filesystem.info(staging_uri, refresh=True)
    except FileNotFoundError:
        raise RuntimeError(
            f"EDC transfer {transfer['@id']} reported COMPLETED, but {staging_uri} "
            "was not found through the active stack's S3 endpoint. Compare the "
            "logged PUT destination and read endpoint with the working standalone "
            "script; ask the provider owner to check the actual PUT response."
        ) from None
    logger.info("Verified staged CSV: uri=%s bytes=%s", staging_uri, object_info.get("size"))
    with filesystem.open(staging_uri, "rb") as source:
        df = pd.read_csv(source)
    return df, {
        "dataset_id": dataset_id, "asset_id": asset_id,
        "provider_connector": provider, "consumer_connector": consumer,
        "agreement_id": agreement_id, "transfer_id": transfer["@id"],
        "staging_uri": staging_uri,
    }


@step(name="load-data", enable_cache=False)
def load_data(
dataset_id: str,
asset_id: str,
) -> Annotated[pd.DataFrame, "raw_dataset"]:
    """Retrieve a raw CSV using the dataspace secret and stack storage credentials."""
    if not dataset_id or not asset_id:
        raise ValueError("dataset_id and asset_id are required.")
    df, provenance = _retrieve_dataspace_dataset(dataset_id, asset_id)
    if df.empty:
        raise ValueError("The retrieved dataset is empty.")
    log_metadata(metadata={
        **provenance, "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]), "columns": list(df.columns),
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
    }, infer_artifact=True)
    logger.info("Loaded dataspace dataset: rows=%d columns=%d", *df.shape)
    return df


@step(name="pre-process", enable_cache=False)
def pre_process(
df: pd.DataFrame,
test_size: float = 0.2,
) -> Tuple[
Annotated[np.ndarray, "train_x"],
Annotated[np.ndarray, "train_y"],
Annotated[np.ndarray, "test_x"],
Annotated[np.ndarray, "test_y"],
]:
    """Prepare next-observation CPU forecasting samples.

    Training targets precede the shared test cutoff to prevent time leakage.
    CPU values above 100 are valid for containers using multiple cores."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    required = {"timestamp", "container_name", "cpu_percent", "memory_usage", "pids"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing telemetry columns: {sorted(missing)}")
    data = df.loc[:, sorted(required)].copy()
    if data.isna().any().any():
        raise ValueError("Required telemetry columns contain missing values.")
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
    if data.duplicated(["container_name", "timestamp"]).any():
        raise ValueError("Expected one observation per container and timestamp.")
    data["cpu_percent"] = pd.to_numeric(
        data["cpu_percent"].astype(str).str.strip().str.removesuffix("%"),
        errors="raise",
    )
    # Extract used memory (e.g. 261MiB) and normalize mixed units to MiB.
    memory = data["memory_usage"].astype(str).str.split("/", n=1).str[0].str.extract(
        r"^\s*(\d+(?:\.\d+)?)\s*(B|kB|KB|MB|GB|TB|KiB|MiB|GiB|TiB)\s*$"
    )
    factors = {"B": 1, "kB": 1000, "KB": 1000, "MB": 1000**2,
               "GB": 1000**3, "TB": 1000**4, "KiB": 1024,
               "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}
    data["memory_used_mib"] = pd.to_numeric(memory[0]) * memory[1].map(factors) / 1024**2
    data["pids"] = pd.to_numeric(data["pids"], errors="raise")
    if not np.isfinite(data[FEATURE_NAMES].to_numpy(dtype=np.float64)).all():
        raise ValueError("Telemetry features must be finite numbers with supported memory units.")

    data = data.sort_values(["container_name", "timestamp"])
    groups = data.groupby("container_name", sort=False)
    data["next_cpu_percent"] = groups["cpu_percent"].shift(-1)
    data["target_timestamp"] = groups["timestamp"].shift(-1)
    times = data["timestamp"].drop_duplicates().sort_values().tolist()
    split_index = int(len(times) * (1 - test_size))
    if not 0 < split_index < len(times):
        raise ValueError("Not enough distinct timestamps for the requested split.")
    cutoff = times[split_index]
    pairs = data.dropna(subset=["next_cpu_percent", "target_timestamp"])
    train_df = pairs.loc[pairs["target_timestamp"] < cutoff]
    test_df = pairs.loc[pairs["timestamp"] >= cutoff]
    if train_df.empty or test_df.empty:
        raise ValueError("The chronological split must produce non-empty training and test sets.")

    train_x = train_df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    train_y = train_df["next_cpu_percent"].to_numpy(dtype=np.float32)
    test_x = test_df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    test_y = test_df["next_cpu_percent"].to_numpy(dtype=np.float32)
    log_metadata(metadata={
        "target": "next observed CPU percent per container",
        "feature_names": FEATURE_NAMES, "feature_count": len(FEATURE_NAMES),
        "split_method": "chronological; training targets precede test period",
        "test_start_utc": cutoff.isoformat(), "test_size": float(test_size),
        "train_rows": len(train_df), "test_rows": len(test_df),
        "dropped_final_rows": int(len(data) - len(pairs)),
        "dropped_boundary_rows": int(len(pairs) - len(train_df) - len(test_df)),
        "median_forecast_seconds": float(
            (pairs["target_timestamp"] - pairs["timestamp"]).dt.total_seconds().median()
        ),
    })
    return train_x, train_y, test_x, test_y


@step(name="train-random-forest", enable_cache=False)
def train(
    train_x: np.ndarray,
    train_y: np.ndarray,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42,
) -> Annotated[
    RandomForestRegressor,
    ArtifactConfig(
        name="trained_random_forest",
        artifact_type=ArtifactType.MODEL,
        tags=["random-forest", "regression", "sklearn", "trained"],
    ),
]:
    """Learn next-observation CPU usage from the numeric telemetry features."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
        # Forests report tree-building progress rather than epochs.
        verbose=2,
    )

    logger.info("Training Random Forest regressor for next-observation CPU usage...")
    model.fit(train_x, train_y)
    logger.info(
        "Model fitting completed: trees=%d features=%d. Returning the model for ZenML artifact persistence.",
        len(model.estimators_),
        model.n_features_in_,
    )

    params = {
        "n_estimators": int(n_estimators),
        "max_depth": max_depth,
        "min_samples_split": int(min_samples_split),
        "min_samples_leaf": int(min_samples_leaf),
        "random_state": int(random_state),
        "n_features": int(train_x.shape[1]),
    }
    log_metadata(metadata={"training_parameters": params}, infer_artifact=True)

    return model


@step(name="evaluate", enable_cache=False)
def evaluate(
    model: RandomForestRegressor,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> Dict[str, float]:
    """Measure held-out error: MAE/RMSE in CPU percentage points, plus R²."""
    predicted = model.predict(test_x)

    rmse = float(np.sqrt(mean_squared_error(test_y, predicted)))
    mae = float(mean_absolute_error(test_y, predicted))
    r2 = float(r2_score(test_y, predicted))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

    log_metadata(
        metadata={
            "metrics": metrics,
            "test_samples": int(len(test_y)),
        }
    )

    logger.info(
        "Evaluation complete | MAE=%.6f RMSE=%.6f CPU percentage points | R2=%.6f",
        mae,
        rmse,
        r2,
    )

    return metrics


@step(name="export-onnx", enable_cache=False)
def export_onnx(
    model: RandomForestRegressor,
) -> Annotated[
    ONNXArtifact,
    ArtifactConfig(
        name="random_forest_onnx",
        artifact_type=ArtifactType.MODEL,
        tags=["random-forest", "onnx", "downloadable"],
    ),
]:
    logger.info("Converting Random Forest to ONNX...")
    num_features = int(model.n_features_in_)
    initial_type = [("float_input", FloatTensorType([None, num_features]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
    )
    import onnx

    onnx.checker.check_model(onnx_model)

    onnx_bytes = onnx_model.SerializeToString()

    log_metadata(
        metadata={
            "format": "onnx",
            "framework": "scikit-learn",
            "onnx_size_bytes": int(len(onnx_bytes)),
            "filename": ONNXMaterializer.FILENAME,
            "feature_names": FEATURE_NAMES,
            "target": "next observed CPU percent per container",
        },
        infer_artifact=True,
    )

    return ONNXArtifact(onnx_bytes)


@step(name="publish-result", enable_cache=False)
def upload_model_and_report(
    metrics: Dict[str, float],
    onnx_artifact: UnmaterializedArtifact,
    run_name: str,
    experiment_name: str,
    dataspace_model_id: str,
    presign_expires: int = 3600,
) -> Dict[str, Any]:
    """Create the result object consumed by the Control API.

    The UnmaterializedArtifact input asks ZenML to pass an artifact reference
    instead of loading the ONNXArtifact bytes through the materializer. This
    step inspects its ID, name, and URI, then opens the serialized file to
    upload a copy to the central Data Lake without deserializing the model.
    This uploads model.onnx only; catalogue registration also requires a
    metadata.json record, which is not produced by this example yet.
    """
    import re

    import boto3
    from botocore.config import Config

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", dataspace_model_id):
        raise ValueError("dataspace_model_id must be a non-empty single directory name.")

    artifact_uri = onnx_artifact.uri
    model_uri = _artifact_file_uri(
        artifact_uri,
        ONNXMaterializer.FILENAME,
    )

    config = Client().get_secret("datalake-model-storage").secret_values
    required_keys = ("DLAKE_ENDPOINT", "DLAKE_ACCESS_KEY", "DLAKE_SECRET_KEY")
    missing_keys = [key for key in required_keys if not config.get(key)]
    if missing_keys:
        raise ValueError(
            "ZenML secret 'datalake-model-storage' is missing required keys: "
            + ", ".join(missing_keys)
        )
    endpoint = config["DLAKE_ENDPOINT"]
    region = config.get("DLAKE_REGION") or "us-east-1"
    bucket = "6g-dali-models"
    key = f"{dataspace_model_id}/model.onnx"
    s3 = boto3.client(
        "s3", endpoint_url=endpoint,
        aws_access_key_id=config["DLAKE_ACCESS_KEY"],
        aws_secret_access_key=config["DLAKE_SECRET_KEY"],
        region_name=region,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    artifact_store = Client().active_stack.artifact_store
    with artifact_store.open(model_uri, "rb") as source:
        s3.upload_fileobj(source, bucket, key, ExtraArgs={"ContentType": "application/onnx"})
    uploaded = s3.head_object(Bucket=bucket, Key=key)
    datalake_uri = f"s3://{bucket}/{key}"
    logger.info("Uploaded ONNX model: %s bytes=%s", datalake_uri, uploaded["ContentLength"])

    presigned_url = _presigned_url(
        artifact_uri=artifact_uri,
        filename=ONNXMaterializer.FILENAME,
        expires=presign_expires,
    )

    result = {
        "pipeline_name": "random_forest_pipeline",
        "run_name": run_name,
        "experiment_name": experiment_name or None,
        "status": "SUCCESS",
        "metrics": metrics,
        "model": {
            "artifact_id": str(onnx_artifact.id),
            "artifact_name": str(onnx_artifact.name),
            "artifact_uri": artifact_uri,
            "format": "onnx",
            "filename": ONNXMaterializer.FILENAME,
            "s3_uri": model_uri if model_uri.startswith("s3://") else None,
            "artifact_store_uri": model_uri,
            "dataspace_model_id": dataspace_model_id,
            "datalake_uri": datalake_uri,
            "datalake_size_bytes": int(uploaded["ContentLength"]),
            "presigned_url": presigned_url,
            "presigned_expires_seconds": int(presign_expires),
        },
    }

    log_metadata(
        metadata={
            "api_result": result,
            "model_artifact_uri": model_uri,
            "model_presigned_url": presigned_url or "",
            "metrics": metrics,
        }
    )

    logger.info("Published result for ONNX artifact: %s", model_uri)
    return result


@pipeline(
    name="random_forest_pipeline",
    settings={
        "docker": docker_settings,
    },
)
def random_forest_pipeline(
    dataset_id: str,
    run_name: str,
    dataspace_model_id: str = "rf-example-001",
    asset_id: str = "d1c036a0-feaa-4b4f-8277-0b3c541d5069",
    experiment_name: str = "",
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 50,
    max_depth: Optional[int] = 10,
    min_samples_split: int = 2,
    min_samples_leaf: int = 2,
    presign_expires: int = 3600,
) -> Dict[str, Any]:
    """Train, evaluate and publish a Random Forest ONNX artifact."""

    df = load_data(dataset_id=dataset_id, asset_id=asset_id)

    train_x, train_y, test_x, test_y = pre_process(
        df=df,
        test_size=test_size,
    )

    model = train(
        train_x=train_x,
        train_y=train_y,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )

    metrics = evaluate(
        model=model,
        test_x=test_x,
        test_y=test_y,
    )

    onnx_model = export_onnx(
        model=model,
    )

    result = upload_model_and_report(
        metrics=metrics,
        onnx_artifact=onnx_model,
        run_name=run_name,
        experiment_name=experiment_name,
        dataspace_model_id=dataspace_model_id,
        presign_expires=presign_expires,
    )

    return result


if __name__ == "__main__":
    random_forest_pipeline(
        dataset_id="f32144c2-2bb5-4b3e-bde9-e7a99270b091",
        asset_id="d1c036a0-feaa-4b4f-8277-0b3c541d5069",
        run_name="rf-demo-001",
        experiment_name="cpu-forecasting",
        n_estimators=50,
        max_depth=10,
        min_samples_leaf=2,
        presign_expires=3600,
    )
