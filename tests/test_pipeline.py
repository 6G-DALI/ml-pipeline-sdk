import inspect
from pathlib import Path

import pytest
from zenml import pipeline, step
from zenml.client import Client

from dali_mlops import Checkpoint, TrainingOutput, dali_pipeline
from dali_mlops.pipeline import docker_settings, unpack
from dali_mlops.spec import parse_pipeline_source

RESERVED_ORDER = ["dataset_id", "dataspace_model_id", "run_name", "asset_id", "experiment_name"]
ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("ZENML_CONFIG_PATH", str(tmp_path / "zenml"))
    monkeypatch.setenv("ZENML_ANALYTICS_OPT_IN", "false")
    Client._reset_instance()
    yield
    Client._reset_instance()


@step
def make_model() -> int:
    return 1


@dali_pipeline
def forest(df, n: int = 3, label: str = "y"):
    return TrainingOutput(df, {"mae": 1.0})


@dali_pipeline(name="Named", dataset=None, base_image="img:1", environment={"EDC_PROVIDER_ID": "other"})
def no_data(epochs: int, lr: float = 0.1):
    return make_model(), {"mae": 1.0}


@dali_pipeline(resumable=True)
def resumable_forest(df, base: Checkpoint | None = None, n: int = 3, extra: int = 5):
    return TrainingOutput(df, {"mae": 1.0}, checkpoint=Checkpoint.from_bytes(b"x"))


@dali_pipeline(dataset=None, resumable=True)
def resumable_no_data(base: Checkpoint | None = None, epochs: int = 1):
    return TrainingOutput(1, {}, checkpoint=Checkpoint.from_bytes(b"x"))


def parameters(pipe):
    return list(inspect.signature(pipe.entrypoint).parameters.values())


def test_signature_lists_the_reserved_parameters_then_the_authors():
    params = parameters(forest)
    assert [p.name for p in params] == RESERVED_ORDER + ["n", "label"]
    assert [p.annotation for p in params[:5]] == [str, str, str, str | None, str]
    assert params[3].default is None and params[4].default == "" and params[5].default == 3


def test_required_author_parameter_precedes_defaulted_ones():
    assert [p.name for p in parameters(no_data)] == ["dataspace_model_id", "run_name", "epochs", "experiment_name", "lr"]


def test_the_pipeline_keeps_the_author_name_for_source_resolution():
    assert forest.name == "forest" and no_data.name == "Named"
    assert forest.entrypoint.__name__ == "forest" and forest.entrypoint.__module__ == __name__
    assert inspect.getsource(forest.entrypoint).lstrip().startswith("@dali_pipeline")


def test_signature_matches_the_static_spec():
    source = (ROOT / "examples" / "random_forest.py").read_text()
    spec = parse_pipeline_source(source)["pipelines"][0]
    from examples.random_forest import random_forest_pipeline as imported

    assert [p.name for p in parameters(imported)] == [p["name"] for p in spec["params"]]
    assert [p.default is inspect.Parameter.empty for p in parameters(imported)] == [p["required"] for p in spec["params"]]


def test_epoch_example_spec():
    spec = parse_pipeline_source((ROOT / "examples" / "epoch_mlp.py").read_text())["pipelines"][0]
    assert [p["name"] for p in spec["params"]] == RESERVED_ORDER + ["base_model_uri", "target", "epochs", "learning_rate"]
    assert spec["sdk"] and spec["dataset_format"] == "csv" and spec["requirements"]


def test_settings_follow_base_image_and_requirements():
    assert forest.configuration.settings == {} and forest.configuration.enable_cache is False
    docker = no_data.configuration.settings["docker"]
    assert (docker.parent_image, docker.skip_build, docker.requirements) == ("img:1", True, None)
    assert no_data.configuration.environment == {"EDC_PROVIDER_ID": "other"}
    built = docker_settings("img:2", ["numpy"])
    assert (built.parent_image, built.skip_build, built.requirements) == ("img:2", False, ["numpy"])
    only_requirements = docker_settings(None, Path("requirements.txt"))
    assert only_requirements.parent_image is None and only_requirements.skip_build is False
    assert docker_settings(None, None) is None and docker_settings(None, []) is None


def test_options_pass_through_and_settings_merge():
    @dali_pipeline(base_image="img:3", settings={"resources": {}}, enable_cache=True)
    def merged(df):
        return df, {}

    assert set(merged.configuration.settings) == {"docker", "resources"} and merged.configuration.enable_cache is True


def test_compile_collects_the_pipeline_parameters_and_steps():
    @step
    def noop(df: int) -> int:
        return df

    @dali_pipeline
    def compiled(df, n: int = 5):
        return TrainingOutput(noop(0), {"mae": 1.0})

    compiled.prepare(dataset_id="d", dataspace_model_id="9d1b6f5e-3c47-4c3a-8d86-6b2f1f0a7e10", run_name="r", n=7)
    snapshot, _, _ = compiled._compile()
    assert set(snapshot.step_configurations) == {"retrieve_csv", "noop", "save_model"}
    assert snapshot.pipeline_configuration.parameters["n"] == 7
    assert snapshot.pipeline_configuration.parameters["dataset_id"] == "d"
    assert snapshot.pipeline_configuration.parameters["asset_id"] is None


@pytest.mark.parametrize(("fmt", "step_name"), [
    ("csv", "retrieve_csv"), ("tsv", "retrieve_tsv"), ("parquet", "retrieve_parquet"), ("json", "retrieve_json"),
    ("jsonl", "retrieve_jsonl"), ("raw", "retrieve_dataset"),
])
def test_each_dataset_format_uses_its_retrieval_step(fmt, step_name):
    @dali_pipeline(dataset=fmt)
    def fmt_pipeline(data):
        return data, {}

    fmt_pipeline.prepare(dataset_id="d", dataspace_model_id="9d1b6f5e-3c47-4c3a-8d86-6b2f1f0a7e10", run_name="r")
    snapshot, _, _ = fmt_pipeline._compile()
    assert set(snapshot.step_configurations) == {step_name, "save_model"}


def test_without_a_dataset_there_is_no_retrieval_step():
    no_data.prepare(dataspace_model_id="9d1b6f5e-3c47-4c3a-8d86-6b2f1f0a7e10", run_name="r", epochs=1)
    snapshot, _, _ = no_data._compile()
    assert set(snapshot.step_configurations) == {"make_model", "save_model"}


@pytest.mark.parametrize("make", [
    lambda: dali_pipeline(dataset="xlsx"),
    lambda: dali_pipeline()(lambda: None),
    lambda: dali_pipeline(lambda df, *rest: None),
    lambda: dali_pipeline(lambda df, **extra: None),
    lambda: dali_pipeline(lambda df, run_name="x": None),
    lambda: dali_pipeline(dataset=None)(lambda asset_id=None: None),
])
def test_invalid_declarations(make):
    with pytest.raises((TypeError, ValueError)):
        make()


def test_unpack():
    assert unpack(TrainingOutput("m", {"a": 1}, [1])) == ("m", {"a": 1}, [1])
    assert unpack(("m", {"a": 1})) == ("m", {"a": 1}, None)
    assert unpack(("m", {"a": 1}, [2])) == ("m", {"a": 1}, [2])
    for bad in ("m", ("m",), ("a", "b", "c", "d"), None):
        with pytest.raises(TypeError, match="TrainingOutput"):
            unpack(bad)


def test_a_plain_zenml_pipeline_still_works(offline):
    @pipeline
    def plain(x: int = 1):
        pass

    assert [p.name for p in parameters(plain)] == ["x"]


def test_resumable_signature_adds_base_model_uri_and_hides_base():
    params = parameters(resumable_forest)
    assert [p.name for p in params] == RESERVED_ORDER + ["base_model_uri", "n", "extra"]
    uri = params[5]
    assert uri.annotation == str | None and uri.default is None
    assert [p.name for p in parameters(resumable_no_data)] == [
        "dataspace_model_id", "run_name", "experiment_name", "base_model_uri", "epochs"]


def test_resumable_signature_matches_the_static_spec():
    source = (ROOT / "examples" / "random_forest.py").read_text()
    spec = parse_pipeline_source(source)["pipelines"][0]
    from examples.random_forest import random_forest_pipeline as imported

    assert spec["resumable"] is True
    assert [p.name for p in parameters(imported)] == [p["name"] for p in spec["params"]]
    assert "base_model_uri" in [p["name"] for p in spec["params"]] and "base" not in [p["name"] for p in spec["params"]]
    assert [p.default is inspect.Parameter.empty for p in parameters(imported)] == [p["required"] for p in spec["params"]]
    mlp = parse_pipeline_source((ROOT / "examples" / "epoch_mlp.py").read_text())["pipelines"][0]
    assert mlp["resumable"] is True and mlp["params"][-4]["name"] == "base_model_uri"


def test_a_resumable_pipeline_compiles_with_the_base_step():
    resumable_forest.prepare(
        dataset_id="d", dataspace_model_id="9d1b6f5e-3c47-4c3a-8d86-6b2f1f0a7e10", run_name="r",
        base_model_uri="s3://6g-dali-models/9d1b6f5e-3c47-4c3a-8d86-6b2f1f0a7e10/checkpoint.joblib",
    )
    snapshot, _, _ = resumable_forest._compile()
    assert set(snapshot.step_configurations) == {"retrieve_csv", "load_base_checkpoint", "save_model"}
    assert snapshot.pipeline_configuration.parameters["base_model_uri"].endswith("checkpoint.joblib")
    assert "base" not in snapshot.pipeline_configuration.parameters


def test_pipelines_written_for_0_2_0_are_unchanged():
    # no resumable flag: no base_model_uri, no base step, the old result shape and the old function call
    assert "base_model_uri" not in [p.name for p in parameters(forest)]
    forest.prepare(dataset_id="d", dataspace_model_id="9d1b6f5e-3c47-4c3a-8d86-6b2f1f0a7e10", run_name="r")
    snapshot, _, _ = forest._compile()
    assert set(snapshot.step_configurations) == {"retrieve_csv", "save_model"}
    assert [p.name for p in parameters(forest)] == RESERVED_ORDER + ["n", "label"]

    @dali_pipeline
    def plain(df, base: int = 1):
        return df, {}

    assert [p.name for p in parameters(plain)] == RESERVED_ORDER + ["base"]


@pytest.mark.parametrize("make", [
    lambda: dali_pipeline(resumable=True)(lambda df: None),
    lambda: dali_pipeline(resumable=True)(lambda df, other=None: None),
    lambda: dali_pipeline(resumable=True)(lambda df, base=None: None),
    lambda: dali_pipeline(resumable=True)(_base_required),
    lambda: dali_pipeline(resumable=True)(_base_str),
    lambda: dali_pipeline(resumable=True)(_base_wrong_default),
    lambda: dali_pipeline(resumable=True)(_base_after_param),
    lambda: dali_pipeline(resumable=True)(_declares_base_model_uri),
    lambda: dali_pipeline(dataset=None, resumable=True)(lambda: None),
])
def test_invalid_resumable_declarations(make):
    with pytest.raises(TypeError):
        make()


def _base_required(df, base: Checkpoint | None):
    pass


def _base_str(df, base: str | None = None):
    pass


def _base_wrong_default(df, base: Checkpoint | None = 1):
    pass


def _base_after_param(df, n: int = 1, base: Checkpoint | None = None):
    pass


def _declares_base_model_uri(df, base: Checkpoint | None = None, base_model_uri: str | None = None):
    pass


def test_a_plain_pipeline_cannot_declare_base_model_uri():
    with pytest.raises(TypeError, match="reserved"):
        dali_pipeline(lambda df, base_model_uri=None: None)


def test_unpack_resumable():
    checkpoint = Checkpoint.from_bytes(b"x")
    assert unpack(TrainingOutput("m", {"a": 1}, [1], checkpoint), True) == ("m", {"a": 1}, [1], checkpoint)
    assert unpack(("m", {"a": 1}, [1], checkpoint), True) == ("m", {"a": 1}, [1], checkpoint)
    assert unpack(("m", {"a": 1}, None, checkpoint), True)[3] is checkpoint
    for bad in (TrainingOutput("m", {}), ("m", {}), ("m", {}, None), "m", None):
        with pytest.raises(TypeError, match="checkpoint"):
            unpack(bad, True)
    # a four-tuple is a resumable-only shape; the checkpoint of a plain pipeline is ignored
    with pytest.raises(TypeError):
        unpack(("m", {}, None, checkpoint))
    assert unpack(TrainingOutput("m", {}, None, checkpoint)) == ("m", {}, None)
