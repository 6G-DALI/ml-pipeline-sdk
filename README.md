# 6G-DALI Python MLOps SDK

The SDK gives ZenML pipeline authors steps for retrieving a Data Space distribution
and publishing an exported model. The ML Service Manager (SM) chooses the dataset,
trained model version UUID and run name. The Meta-Orchestrator (MO) runs the
pipeline on the selected stack. ISI and KUL use Kubeflow and a SeaweedFS
artifact store. Staged datasets and ZenML artifacts live on the testbed.
Published models live in the central Data Lake, with separate S3 credentials.

## Install on WSL or Linux

Use Python 3.12 and the `venv` module. On Ubuntu 24.04, install them with
`sudo apt install git python3.12 python3.12-venv`. The URL below is the configured
Git remote. Public clone access has not been verified. Use the URL supplied by
your maintainer if your account needs access.

```bash
git clone https://github.com/6G-DALI/ml-pipeline-sdk.git
cd ml-pipeline-sdk
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

For development, install the package in editable mode with the example and
test dependencies.

```bash
python -m pip install -e '.[example,test,release]'
python -m pytest -q
```

The distribution is named `6g-dali-mlops` and imports as `dali_mlops`.
The intended PyPI command after maintainer publication is
`python -m pip install 6g-dali-mlops`. This work does not publish a PyPI release.

The local verification baseline is WSL2 with Python 3.12.3 and ZenML 0.93.3.
The supplied reference identifies a prebuilt runtime image but gives no Python
or ZenML version. Verify the image, MO client and ZenML server compatibility
before deploying. This repository contains no license file. Maintainers need to
supply the license, confirm repository access, and arrange PyPI ownership and
publishing credentials before publication.

## Public steps

```python
from dali_mlops import (
    DatasetFile, ExportedModel, retrieve_dataset, retrieve_csv, retrieve_tsv,
    retrieve_parquet, retrieve_json, retrieve_jsonl, save_model,
)
from zenml import pipeline, step
```

| API | v0.1 behavior | Output |
| --- | --- | --- |
| `retrieve_dataset(dataset_id, asset_id=None)` | Stage any format as bytes | `DatasetFile` |
| `retrieve_csv(dataset_id, asset_id=None)` | Parse comma-separated data with pandas | `DataFrame` |
| `retrieve_tsv(dataset_id, asset_id=None)` | Parse tab-separated data with pandas | `DataFrame` |
| `retrieve_parquet(dataset_id, asset_id=None)` | Raises `NotImplementedError` before I/O | Planned `DataFrame` |
| `retrieve_json(dataset_id, asset_id=None)` | Raises `NotImplementedError` before I/O | Planned decoded JSON value |
| `retrieve_jsonl(dataset_id, asset_id=None)` | Raises `NotImplementedError` before I/O | Planned tabular `DataFrame` |
| `save_model(model, metrics, dataspace_model_id, run_name, ...)` | Conditionally upload and verify an exported model | Result dict |

The raw step works for Parquet, JSON and JSON Lines too. An example downstream
step reads bytes via the worker's stack connection.

```python
from dali_mlops import DatasetFile, retrieve_dataset
from zenml import pipeline, step

@step
def file_size(dataset: DatasetFile) -> int:
    with dataset.open("rb") as source:
        return sum(len(chunk) for chunk in iter(lambda: source.read(1024 * 1024), b""))

@pipeline
def inspect_file(dataset_id: str, asset_id: str | None = None) -> int:
    return file_size(retrieve_dataset(dataset_id, asset_id))
```

The portable reference carries the filename, media type when known, size, S3
URI, dataset ID, asset ID, agreement ID and transfer ID. Workers need access to
the same staging bucket. CSV and TSV steps use the same staging path and accept
the same arguments. Parsing and domain validation after loading belong to the
author. Retrieval and publication steps disable ZenML caching. Imports
and pipeline definitions require no storage or connector setup.

## Write a training pipeline

A compact example uses a fitted scikit-learn model and explicit ONNX input schema.
Replace `input` and `target` with your own columns.

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from dali_mlops import ExportedModel, retrieve_csv, save_model
from dali_mlops.onnx import export_sklearn_onnx
from zenml import pipeline, step

@step
def fit_export(table: pd.DataFrame) -> ExportedModel:
    estimator = LinearRegression().fit(table[["input"]].to_numpy(), table["target"])
    return export_sklearn_onnx(estimator, input_name="features", feature_count=1)

@pipeline
def train_pipeline(
    dataset_id: str, dataspace_model_id: str, run_name: str, asset_id: str | None = None,
) -> dict:
    model = fit_export(retrieve_csv(dataset_id, asset_id))
    return save_model(model, {}, dataspace_model_id, run_name)
```

For a TSV input use `retrieve_tsv(dataset_id, asset_id)`. Your export step can
instead call `ExportedModel.from_file(path, "application/onnx")` or
`ExportedModel.from_bytes(data, "model.bin", "application/octet-stream")`.
The SDK materializer persists the bytes and metadata for the publication worker.
The current implementation holds exported model bytes in memory and uses a
single S3 PUT for publication. Choose an export size that fits worker memory and
the storage service's single-PUT limit. Publication by itself needs no
scikit-learn, ONNX or skl2onnx dependencies.

## Run the worked example

`examples/random_forest.py` uses the current CPU percentage, used memory in
MiB, and process count to predict the next observed CPU percentage for each
container. The test features are at or after a shared chronological cutoff.
Training targets must precede that cutoff, so a pair crossing it is discarded.
Evaluation reports MAE and RMSE in CPU percentage points and R². The validated
ONNX expects a float32 tensor named `telemetry` of shape `[batch, 3]` in
the order `cpu_percent`, `memory_used_mib`, `pids`.

Run a genuine local ZenML pipeline with the synthetic telemetry fixture, mocked
Data Space boundary, and in-memory S3 staging and publication. This requires
the `example` and `test` extras and uses an isolated local ZenML database.

```bash
python -m pytest -q tests/test_example.py
```

For a real stack, log in to the maintainer-provided ZenML server, activate the
target stack, and run the example from the repository root. The distribution ID
is optional when the dataset has exactly one fetchable distribution. Supply a
fresh trained version UUID from the SM for each independent run.

```bash
zenml login "$ZENML_SERVER_URL"
zenml stack set "$STACK_NAME"
python -m examples.run \
  --dataset-id "$DATASET_ID" \
  --dataspace-model-id "$MODEL_VERSION_UUID" \
  --run-name "$RUN_NAME" \
  --experiment-name cpu-forecasting \
  --runtime-image "$KUBEFLOW_RUNTIME_IMAGE"
```

Set optional `--asset-id "$ASSET_ID"` when selecting a distribution.
The example uses a prebuilt image with `skip_build=True` when
`--runtime-image` is set. Install the wheel with its `example` extra into that
image, using the runtime's Python, along with the configured stack's ZenML
integrations. Include the example source in the MO's pipeline source snapshot
or image as required by the deployment. For an image build context containing
the checked wheel, the Dockerfile can use:

```dockerfile
COPY dist/6g_dali_mlops-0.1.0-py3-none-any.whl /opt/wheels/
RUN python -m pip install '/opt/wheels/6g_dali_mlops-0.1.0-py3-none-any.whl[example]' \
    && python -m pip check
```

The worker needs a compatible S3 artifact store integration (`s3fs` for the
ZenML S3 store) and the Kubeflow integration in its base image. Check the
resolved versions in the intended image. Installing in a workstation virtual
environment alone cannot make the SDK available to Kubeflow workers.

## Configure the worker

Set the following in the environment of the relevant worker pods. Put
credentials in ZenML secrets and the stack's artifact store connection.

| Variable | Meaning |
| --- | --- |
| `PIVEAU_HUB_URL` | Catalogue base such as `https://catalog.example/hub` |
| `EDC_PROVIDER_URL` | Provider connector base or DSP URL override. Defaults to the distribution's `dcat:accessURL` |
| `EDC_CONSUMER_URL` | Consumer connector base such as `https://consumer.example`. Defaults to the provider base for the demo |
| `EDC_PROVIDER_ID` | Provider participant ID |
| `EDC_API_KEY_HEADER` | Consumer management header, default `X-Api-Key` |
| `DALI_TRANSFER_MODE` | `push` by default, or `pull` for a compatible provider and asset |
| `DALI_STAGING_BUCKET` | Bucket on the artifact store connection, defaults to the stack's bucket. The demo uses `datasets` |
| `DALI_STAGING_SIGN_ENDPOINT` | Optional S3 endpoint reachable from the provider, used when signing the push PUT |
| `DALI_POLL_TIMEOUT`, `DALI_POLL_INTERVAL` | Seconds per polling phase and between probes, defaults 120 and 1 |
| `DALI_REQUEST_TIMEOUT`, `DALI_DOWNLOAD_TIMEOUT` | HTTP request/read timeout and streamed pull deadline in seconds, defaults 30 and 300 |
| `DALI_PRESIGN_SECONDS` | Push PUT URL lifetime in seconds, default 900 |

Create ZenML secret `dataspace` with `EDC_API_KEY=<consumer-management-key>`.
Create `datalake-model-storage` with
`DLAKE_ENDPOINT=https://datalake.example`, `DLAKE_ACCESS_KEY=<access-key>`,
`DLAKE_SECRET_KEY=<secret-key>`, and optional `DLAKE_REGION` (default
`us-east-1`). Grant the publication worker conditional PUT and GET access to
the `6g-dali-models` bucket. Verify that the Data Lake honors atomic
`If-None-Match: *` writes before rollout. The SDK uses SigV4 and path-style S3
and confirms the SHA-256 by reading the object back. Identical retries succeed.
Different bytes at the same key raise a conflict.

In push mode (`PresignedHttpData-PUSH`) the worker signs a fresh staging object
key and the provider PUTs to it. The provider must be able to resolve and
reach the signed endpoint, including through any proxy. Configure the
provider-facing endpoint before signing because the host is part of the
signature. Workers can keep using the internal endpoint from the artifact
store for HEAD and GET. In pull mode (`PresignedUrl-PULL`) the worker waits
for the consumer's EDR and must be able to GET the provider's signed endpoint.
It streams the body through temporary worker disk into staging. Size worker
ephemeral storage for the largest expected pull. A separate consumer needs
its own management base and key, as well as federation connectivity to the
provider's DSP endpoint. Use the connector participant ID for that provider.

## From the SM trigger to a registered model

Example SM run parameters for the named Random Forest pipeline. This is the
pipeline parameter object, with the MO request envelope supplied by the
deployment.

```json
{
  "dataset_id": "telemetry-dataset-id",
  "asset_id": "telemetry-csv-asset-id",
  "dataspace_model_id": "96e968f1-f414-40d0-b4dd-8a37897edf74",
  "run_name": "rf-demo-001",
  "experiment_name": "cpu-forecasting",
  "n_estimators": 50
}
```

Representative SDK result, also written to ZenML step metadata as
`api_result` and emitted in step logs.

```json
{
  "run_name": "rf-demo-001",
  "experiment_name": "cpu-forecasting",
  "status": "SUCCESS",
  "metrics": {"mae": 2.3, "model-rmse": 3.1, "r2": 0.87},
  "model": {
    "dataspace_model_id": "96e968f1-f414-40d0-b4dd-8a37897edf74",
    "filename": "model.onnx",
    "datalake_uri": "s3://6g-dali-models/96e968f1-f414-40d0-b4dd-8a37897edf74/model.onnx",
    "datalake_size_bytes": 2560,
    "media_type": "application/onnx",
    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  }
}
```

Final metric names remain as supplied, with `rmse` mapped to `model-rmse`.
For example, `mae`, `r2` and custom names like `latency_ms` are preserved.
Supplying both RMSE names fails. Metric values must be finite numbers.
Optional `metric_history` takes a list of metric maps and emits
`metrics_history`. Random Forest has no epoch history. The expected MO mapping
is SDK `metrics` to FastAPI `metrics.final`, and optional `metrics_history`
to `metrics.history`. The supplied reference contains the producer only.
The MO result reader, live API mapping, and any optional history contract
remain to be verified.

Use the result's S3 URI and authorized Data Lake credentials to locate the
uploaded model immediately. After the run succeeds, the SM writes
`metadata.json` beside the model, including `trained_on`, metrics and
`previous_version` when applicable. SPW's monitor registers the record and
distribution. Once registered, locate the trained version in the Data Space
catalogue, select its connector-backed distribution using its `dali:assetId`
and `dcat:accessURL`, then negotiate and transfer it through the consumer EDC
using the registered provider/asset and a supported transfer mode. Use the
returned ONNX bytes for downstream deployment. The SDK finishes at verified
upload. Dataset publication, catalogue metadata writing, SM orchestration,
deployment and TAI integration are outside this release.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Multiple distributions | Pass an `asset_id` listed in the selection error. Use the ID from the chosen dataset record |
| Negotiation fails | Check the provider participant ID and DSP URL, consumer key and policy conditions. Inspect the connector's negotiation status in its operator interface |
| Transfer starts but EDR is missing | Check that the provider and asset support pull and that the consumer's EDR store receives the data address. Adjust the poll timeout for delayed delivery |
| Push completes but staging is missing or unreadable | Verify the bucket, worker HEAD/GET permissions and provider reachability of the initially signed endpoint. Check that both endpoints refer to the same bucket |
| Upload succeeds but the catalogue record is missing | Use the result URI to confirm the upload, then check SM metadata publication and the SPW monitor |
| Published version conflicts | Keep the existing model and allocate a fresh version UUID for a different export |

## Build and verify

```bash
python -m pip check
python -m build
python -m twine check --strict dist/*
```

CI runs the offline tests, builds the source distribution and wheel, and
installs the wheel into a clean environment outside the source checkout to
verify public imports. Live acceptance is an SM-triggered ISI run with metrics
visible in the SM and a registered ONNX successfully retrieved through EDC.
