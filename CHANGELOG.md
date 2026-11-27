# Changelog

## 0.3.0

- Resume from a previous version: `@dali_pipeline(resumable=True)`. The function takes
  `base: Checkpoint | None = None` after the dataset and returns `TrainingOutput(..., checkpoint=...)`.
- New reserved parameter `base_model_uri` (run contract v4), declared only by resumable pipelines. The decorator
  downloads the checkpoint from the Data Lake and verifies size and SHA-256 (`load_base`).
- Results of resumable pipelines gain `checkpoint` and `base_model_uri`. Publication order: checkpoint, ONNX,
  result. A checkpoint with the same file stem as the model is rejected. A checkpoint above 500 MiB logs a warning.
- `Checkpoint` type and its materializer; helpers `dump_joblib`, `load_joblib`, `dump_torch_state`,
  `load_torch_state`.
- Spec parser: `resumable` flag in the pipeline output, `base_model_uri` in `RESERVED` (new `resumable` key on
  each entry), `reserved_params(has_dataset, resumable)`.
- Examples: `random_forest.py` is resumable (`extra_trees`, `warm_start`), `epoch_mlp.py` is a torch network
  with a `state_dict` checkpoint.
- Pipelines written for 0.2.0 are unchanged.

## 0.2.0

- `@dali_pipeline`, static spec parser, Data Space metric names, ZenML 0.94.6.
