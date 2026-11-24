import json
from pathlib import Path
import subprocess
import sys

import pytest

from dali_mlops.spec import RESERVED_NAMES, parse_pipeline_source

FIXTURES = Path(__file__).parent / "fixtures"
RESERVED_ORDER = ["dataset_id", "dataspace_model_id", "run_name", "asset_id", "experiment_name"]
HYPERPARAMETERS = [
    "test_size", "random_state", "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
]


def parse(name):
    return parse_pipeline_source((FIXTURES / name).read_text())


def only(result):
    assert result["errors"] == []
    assert len(result["pipelines"]) == 1
    return result["pipelines"][0]


def names(pipeline):
    return [param["name"] for param in pipeline["params"]]


def by_name(pipeline):
    return {param["name"]: param for param in pipeline["params"]}


def test_importing_the_parser_does_not_import_zenml():
    code = "import sys, dali_mlops.spec; assert 'zenml' not in sys.modules and 'pandas' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_v01_example():
    pipeline = only(parse("v01_random_forest.py"))
    assert (pipeline["name"], pipeline["function"], pipeline["sdk"]) == ("random_forest_pipeline",) * 2 + (True,)
    assert pipeline["dataset_format"] is None
    assert pipeline["base_image"] is None and pipeline["requirements"] is None and pipeline["environment"] == {}
    assert names(pipeline) == ["dataset_id", "dataspace_model_id", "run_name", "asset_id", "experiment_name"] + HYPERPARAMETERS
    params = by_name(pipeline)
    assert params["dataset_id"] == {"name": "dataset_id", "type": "str", "required": True, "default": None, "reserved": True}
    assert params["asset_id"]["type"] == "str | None" and params["asset_id"]["default"] is None
    assert params["max_depth"] == {"name": "max_depth", "type": "int | None", "required": False, "default": 10, "reserved": False}
    assert {n for n, p in params.items() if p["reserved"]} == RESERVED_NAMES


def test_mo_team_pipeline():
    pipeline = only(parse("mo_team_sdk_pipeline.py"))
    assert pipeline["base_image"] == "nmgrigor/zenml-kubeflow-runtime:v6"
    assert pipeline["skip_build"] is True and pipeline["requirements"] is None
    assert pipeline["environment"] == {
        "PIVEAU_HUB_URL": "https://dataspace.6gdali.eu", "EDC_PROVIDER_ID": "provider-facility",
        "EDC_PROVIDER_URL": "https://edc.dataspace.6gdali.eu", "EDC_CONSUMER_URL": "https://edc.dataspace.6gdali.eu",
    }
    assert names(pipeline)[:5] == RESERVED_ORDER[:1] + ["dataspace_model_id", "run_name", "asset_id", "experiment_name"]
    assert pipeline["sdk"] is True


def test_pre_sdk_pipeline_follows_the_docker_settings_constant():
    result = parse("pre_sdk_pipeline.py")
    pipeline = only(result)
    assert pipeline["sdk"] is False
    assert pipeline["base_image"] == "nmgrigor/zenml-kubeflow-runtime:v5" and pipeline["skip_build"] is True
    params = by_name(pipeline)
    assert params["dataspace_model_id"]["default"] == "rf-example-001"
    assert params["max_depth"]["type"] == "Optional[int]"
    assert params["presign_expires"]["default"] == 3600
    assert params["presign_expires"]["reserved"] is False
    assert result["warnings"] == []


def test_dali_pipeline_adds_the_reserved_parameters():
    pipeline = only(parse_pipeline_source('''
from dali_mlops import dali_pipeline

@dali_pipeline(name="Forest", dataset="parquet", base_image="img:1", requirements=["scikit-learn==1.5.2"],
               environment={"EDC_PROVIDER_ID": "other"})
def forest(table, n_estimators: int = 50, label: str = "y"):
    pass
'''))
    assert (pipeline["name"], pipeline["function"], pipeline["sdk"], pipeline["dataset_format"]) == (
        "Forest", "forest", True, "parquet")
    assert names(pipeline) == RESERVED_ORDER + ["n_estimators", "label"]
    assert pipeline["base_image"] == "img:1" and pipeline["requirements"] == ["scikit-learn==1.5.2"]
    assert pipeline["skip_build"] is False
    assert pipeline["environment"] == {"EDC_PROVIDER_ID": "other"}
    assert all(p["reserved"] for p in pipeline["params"][:5]) and not any(p["reserved"] for p in pipeline["params"][5:])


def test_dali_pipeline_defaults_and_no_dataset():
    default = only(parse_pipeline_source("@dali_pipeline\ndef a(df, k: int = 1): pass"))
    assert default["name"] == "a" and default["dataset_format"] == "csv" and "table" not in names(default)
    assert names(default) == RESERVED_ORDER + ["k"]
    none = only(parse_pipeline_source("@dali_pipeline(dataset=None)\ndef b(k: int = 1): pass"))
    assert none["dataset_format"] is None
    assert names(none) == ["dataspace_model_id", "run_name", "experiment_name", "k"]


def test_required_author_parameter_is_placed_before_defaulted_ones():
    pipeline = only(parse_pipeline_source("@dali_pipeline\ndef a(df, epochs: int, lr: float = 0.1): pass"))
    assert names(pipeline) == ["dataset_id", "dataspace_model_id", "run_name", "epochs", "asset_id", "experiment_name", "lr"]


def test_base_image_requirements_file_and_constants():
    pipeline = only(parse_pipeline_source('''
IMAGE = "img:" + "2"
REQ = "requirements.txt"
@dali_pipeline(base_image=IMAGE, requirements=REQ)
def a(df): pass
'''))
    assert (pipeline["base_image"], pipeline["requirements"], pipeline["skip_build"]) == ("img:2", "requirements.txt", False)


def test_inline_docker_settings_and_attribute_decorators():
    pipeline = only(parse_pipeline_source('''
import zenml
from zenml.config import DockerSettings

@zenml.pipeline(name="P", settings={"docker": DockerSettings(parent_image="x:1", requirements=["a", "b"])})
def p(n: int = -1): pass
'''))
    assert (pipeline["name"], pipeline["base_image"], pipeline["requirements"]) == ("P", "x:1", ["a", "b"])
    assert pipeline["params"][0]["default"] == -1


def test_plain_pipeline_without_settings_importing_the_sdk_is_sdk():
    pipeline = only(parse_pipeline_source("from dali_mlops import save_model\n@pipeline\ndef p(): pass"))
    assert pipeline["sdk"] is True and pipeline["params"] == [] and pipeline["base_image"] is None


def test_warnings_for_unreadable_values():
    result = parse_pipeline_source('''
import os
@dali_pipeline(dataset="xlsx", environment=dict(A=os.environ["A"]))
def p(df, a: int = int("1"), *args, **kwargs): pass
''')
    assert result["errors"] == []
    text = " ".join(result["warnings"])
    for fragment in ("dataset must be one of", "default of 'a'", "*args", "environment"):
        assert fragment in text


def test_reserved_names_in_an_sdk_body_are_dropped_with_a_warning():
    result = parse_pipeline_source("@dali_pipeline\ndef p(df, run_name: str = 'x', k: int = 1): pass")
    assert names(only(result)) == RESERVED_ORDER + ["k"]
    assert "reserved" in result["warnings"][0]


@pytest.mark.parametrize(("source", "message"), [
    ("def broken(:", "Syntax error at line 1"),
    ("def plain(): pass", "No pipeline found"),
    ("@step\ndef s(): pass", "No pipeline found"),
    ("", "No pipeline found"),
])
def test_errors(source, message):
    result = parse_pipeline_source(source)
    assert result["pipelines"] == [] and message in result["errors"][0]


def test_several_pipelines_warn():
    result = parse_pipeline_source("@pipeline\ndef a(): pass\n@pipeline\ndef b(): pass")
    assert [p["name"] for p in result["pipelines"]] == ["a", "b"] and result["warnings"]


def test_code_is_never_executed(tmp_path):
    marker = tmp_path / "ran"
    parse_pipeline_source(f"open({str(marker)!r}, 'w')\n@pipeline\ndef p(): pass")
    assert not marker.exists()


def test_result_is_json_serialisable():
    for name in ("v01_random_forest.py", "mo_team_sdk_pipeline.py", "pre_sdk_pipeline.py"):
        json.dumps(parse(name))
