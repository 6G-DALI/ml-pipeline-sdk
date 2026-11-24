# 6G-DALI Python MLOps SDK

A pipeline author writes only the ML logic. The SDK retrieves the dataset through the Data Space, publishes
the trained model to the Data Lake and reports metrics in the Data Space vocabulary. The ML Service Manager
(SM) chooses the dataset, the trained model version UUID and the run name, and the Meta-Orchestrator (MO)
runs the pipeline on the selected ZenML stack.

The distribution is `6g-dali-mlops` and imports as `dali_mlops`. Version 0.2 targets ZenML 0.94.6 (the
MO server version), Python 3.12.

## Install

```bash
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install .                    # the SDK
python -m pip install -e '.[example,test,release]'   # development
```

Extras: `parquet` (pyarrow), `torch` (`export_torch_onnx`), `example` (scikit-learn, skl2onnx, onnx).

## Write a pipeline in 20 lines

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from zenml import step

from dali_mlops import ExportedModel, TrainingOutput, dali_pipeline
from dali_mlops.onnx import export_sklearn_onnx


@step
def fit(table: pd.DataFrame, target: str) -> ExportedModel:
    estimator = LinearRegression().fit(table.drop(columns=[target]).to_numpy(), table[target])
    return export_sklearn_onnx(estimator, input_name="features", feature_count=table.shape[1] - 1)


@dali_pipeline(dataset="csv", base_image="nmgrigor/zenml-kubeflow-runtime:v6")
def linear_pipeline(table: pd.DataFrame, target: str = "target"):
    return TrainingOutput(fit(table, target), {"mae": 0.0})
```

`@dali_pipeline` parameters:

| Parameter | Meaning |
| --- | --- |
| `name` | Pipeline name, default the function name |
| `dataset` | `"csv"` (default), `"tsv"`, `"parquet"`, `"json"`, `"jsonl"`, `"raw"` or `None` |
| `base_image` | Parent image the steps run in |
| `requirements` | List of requirements, or a path to a requirements file |
| `environment` | Overrides for the Data Space settings below, as a dict |
| `enable_cache` | Default `False` |

What the decorator does:

- Adds the run contract parameters to the pipeline signature, so the MO's definitions list them. The function
  declares only its hyperparameters.
- Calls the retrieval step for the declared format and passes the loaded dataset as the **first argument**
  of your function (omitted with `dataset=None`).
- Calls `save_model` on what the function returns: `TrainingOutput(model, metrics, metric_history=None)`
  or a `(model, metrics)` tuple. `model` is an `ExportedModel`.
- Sets `DockerSettings(parent_image=base_image, requirements=requirements, skip_build=not requirements)`.
  With a base image and no requirements the image is used as is. With requirements ZenML builds an image,
  which needs an image builder and a container registry on the stack. With neither, no Docker settings are
  added.
- Keeps `__name__`, `__module__` and the source of your function, so ZenML resolves and snapshots the
  pipeline as the author's module attribute.

Your function cannot declare the reserved names or use `*args`/`**kwargs`.

### How the signature is built, and the fallback

The wrapper sets `__signature__` on the function it hands to `zenml.pipeline`. This was checked on ZenML
0.94.6: `inspect.signature(pipeline.entrypoint)`, the compiled parameters and local runs all see the reserved
parameters. A parser that **imports** the module therefore sees them. A parser that reads the source
**statically** (AST) sees only the function you wrote. `dali_mlops.spec` handles `@dali_pipeline` and adds
the reserved parameters itself, and the MO's own definitions are authoritative after registration.

If an MO cannot cope with a rewritten signature, use the explicit form: a plain `@pipeline` that declares the
reserved parameters itself and calls the steps by hand. This is what v0.1 required, and it still works:

```python
@pipeline(settings={"docker": DockerSettings(parent_image="img", skip_build=True)})
def explicit(dataset_id: str, dataspace_model_id: str, run_name: str, asset_id: str | None = None,
             experiment_name: str = "", n_estimators: int = 50) -> dict:
    table = retrieve_csv(dataset_id, asset_id)
    ...
    return save_model(model, metrics, dataspace_model_id, run_name, experiment_name)
```

## Run contract

The SM fills the reserved parameters. The author never does.

| Parameter | Filled with | Notes |
| --- | --- | --- |
| `dataset_id` | Piveau dataset id | Present when `dataset` is not `None` |
| `asset_id` | Chosen distribution, else omitted (`None`) | The SDK takes the only fetchable distribution. With several, the error lists them |
| `dataspace_model_id` | UUID of the future trained version | The model is written to `<uuid>/<filename>` with `If-None-Match: *` |
| `run_name` | SM process id | Also the MO run name |
| `experiment_name` | Training request name | Optional, default `""` |
| hyperparameters | `training_config.hyperparameters` | Only the names the pipeline declares |

Signature order: required reserved parameters, required author parameters, then the defaulted ones.

## Results contract

The `save_model` step returns the `api_result`, writes it to the step metadata as `api_result` and logs it.
The MO serves it unchanged at `GET /runs/{run}/results`.

```json
{
  "run_name": "rf-demo-001",
  "experiment_name": "cpu-forecasting",
  "status": "SUCCESS",
  "metrics": {"model-mae": 2.3, "model-rmse": 3.1, "model-r2": 0.87},
  "metrics_history": [{"epoch": 0, "loss": 1.2}, {"epoch": 1, "loss": 0.8}],
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

`metrics_history` appears only when the pipeline returns a history: one dict per epoch, with an integer
`epoch`. `experiment_name` is `null` when empty. The consumer checks that `datalake_uri` is
`s3://<bucket>/<dataspace_model_id>/<filename>`.

### Metric names

Names are mapped to Data Space names, in the final metrics and in every history entry. Other names are
lower-cased and kebab-cased (`Train Loss` becomes `train-loss`). Two names that map to the same result are an
error. Values must be finite numbers.

| You write | Reported as |
| --- | --- |
| `accuracy` | `model-accuracy` |
| `f1`, `f1_score` | `model-f1` |
| `rmse` | `model-rmse` |
| `mae` | `model-mae` |
| `r2` | `model-r2` |
| `latency_ms`, `inference_latency_ms` | `inference-latency-ms` |
| `model_size_mb` | `model-size-mb` |

## Configuration

Resolution order for each setting: step or worker environment, then the ZenML secret, then the SDK default.
The defaults are the 6G-DALI production values, so a pipeline needs no configuration on the production stack.

| Setting | Default | Secret |
| --- | --- | --- |
| `PIVEAU_HUB_URL` | `https://dataspace.6gdali.eu` | `dataspace` |
| `EDC_PROVIDER_URL` | `https://edc.dataspace.6gdali.eu` | `dataspace` |
| `EDC_CONSUMER_URL` | `https://edc.dataspace.6gdali.eu` | `dataspace` |
| `EDC_PROVIDER_ID` | `provider-facility` | `dataspace` |
| `DLAKE_BUCKET` | `6g-dali-models` | `datalake-model-storage` |

Keys exist only in secrets. `dataspace` holds `EDC_API_KEY`. `datalake-model-storage` holds
`DLAKE_ENDPOINT`, `DLAKE_ACCESS_KEY`, `DLAKE_SECRET_KEY` and optional `DLAKE_REGION` (default `us-east-1`).
Extra keys in these secrets are allowed. The `environment` argument of `@dali_pipeline` sets the step
environment, which wins over everything else.

Tuning, all optional environment variables: `EDC_API_KEY_HEADER` (default `X-Api-Key`), `DALI_TRANSFER_MODE`
(`push` default, or `pull`), `DALI_STAGING_BUCKET`, `DALI_STAGING_SIGN_ENDPOINT`, `DALI_POLL_TIMEOUT` (120),
`DALI_POLL_INTERVAL` (1), `DALI_REQUEST_TIMEOUT` (30), `DALI_DOWNLOAD_TIMEOUT` (300), `DALI_PRESIGN_SECONDS`
(900).

Staging and pull/push behave as in v0.1: in push mode the provider PUTs to a signed staging URL on the artifact
store, in pull mode the worker streams the provider's signed endpoint into staging. Publication uses a
conditional single PUT with SigV4 and path-style S3 and verifies the SHA-256 by reading the object back.
Identical retries succeed and different bytes at the same key raise a conflict.

## Retrieval and export helpers

| API | Output |
| --- | --- |
| `retrieve_dataset(dataset_id, asset_id=None)` | `DatasetFile`, any format, `.open("rb")` |
| `retrieve_csv`, `retrieve_tsv` | `DataFrame` |
| `retrieve_parquet` | `DataFrame` (needs the `parquet` extra) |
| `retrieve_jsonl` | `DataFrame`, one record per line |
| `retrieve_json` | the decoded JSON value |
| `export_sklearn_onnx(model, input_name=, feature_count=)` | `ExportedModel`, `model.onnx` |
| `export_torch_onnx(model, example_input)` (`torch` extra) | `ExportedModel`, dynamic batch axis |
| `ExportedModel.from_file(path)`, `.from_bytes(data)` | any format, default `model.onnx`, `application/onnx` |

`ExportedModel` holds the bytes in memory and publication is a single PUT. Choose a size that fits the worker
and the storage service.

## Pipeline spec parser

`dali_mlops/spec.py` is standard library only. It never executes code and does not import ZenML or the rest
of the package, so the SM vendors this single file.

```python
from dali_mlops.spec import parse_pipeline_source

parse_pipeline_source(text)
# {"pipelines": [{"name", "function", "sdk", "dataset_format",
#                 "params": [{"name", "type", "required", "default", "reserved"}],
#                 "base_image", "requirements", "skip_build", "environment"}],
#  "errors": [str], "warnings": [str]}
```

It reads `@dali_pipeline(...)` and plain `@pipeline(settings={"docker": DockerSettings(...)}, environment=...)`,
literal values, simple module constants and `DockerSettings` held in a constant. `errors` holds syntax errors
and "no pipeline found". `warnings` holds unreadable values. `RESERVED_NAMES` and `reserved_params()` expose the
run contract.

## Examples

`examples/random_forest.py` forecasts the next CPU percentage per container from telemetry, and
`examples/epoch_mlp.py` reports per-epoch metrics. Both need only their ML logic. Run them locally against a
mocked Data Space and in-memory S3:

```bash
python -m pytest -q tests/test_example.py
```

On a real stack, log in, activate the stack and run `python -m examples.run --dataset-id ... --dataspace-model-id
<fresh uuid> --run-name ...`.

## Migration from 0.1

- ZenML is pinned to 0.94.6 (was 0.93.3).
- Replace the hand-written `@pipeline` signature and the `retrieve_*`/`save_model` calls with `@dali_pipeline`.
  The explicit form still works.
- `PIVEAU_HUB_URL` and `EDC_PROVIDER_ID` are no longer required in the worker environment.
- Metric names change. `rmse` was the only alias. Every other name is now kebab-cased and the common aliases
  map to Data Space names, so `mae` is reported as `model-mae` and `r2` as `model-r2`.
- `retrieve_parquet`, `retrieve_json` and `retrieve_jsonl` are implemented.
- `ExportedModel.from_file` and `from_bytes` default to `model.onnx` and `application/onnx`.
- `import dali_mlops` no longer imports ZenML until a name is used.

## Build and verify

```bash
python -m pip check
python -m pytest -q
python -m build
python -m twine check --strict dist/*
```

CI runs the same and installs the wheel into a clean environment to verify the public imports.
