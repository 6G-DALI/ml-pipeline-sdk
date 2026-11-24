from io import BytesIO

import pandas as pd
import pytest

from dali_mlops import (
    DatasetFile, SDKError, retrieve_csv, retrieve_dataset, retrieve_json, retrieve_jsonl, retrieve_parquet,
    retrieve_tsv,
)
from dali_mlops import retrieval

REFERENCE = DatasetFile("file", None, "demo", "asset", "s3://datasets/object", "agreement", "transfer", 8)


def serve(monkeypatch, payload):
    called = []

    def fetch(dataset, asset):
        called.append((dataset, asset))
        return REFERENCE

    monkeypatch.setattr(retrieval, "fetch_dataset", fetch)
    monkeypatch.setattr(DatasetFile, "open", lambda self, mode: BytesIO(payload))
    return called


@pytest.mark.parametrize(("reader", "delimiter"), [(retrieve_csv, b","), (retrieve_tsv, b"\t")])
def test_delimited_readers(monkeypatch, reader, delimiter):
    called = serve(monkeypatch, delimiter.join([b"x", b"y\n1", b"2\n"]))
    assert reader.entrypoint("demo", "asset").to_dict("records") == [{"x": 1, "y": 2}]
    assert retrieve_dataset.entrypoint("demo", "asset") is REFERENCE
    assert called == [("demo", "asset"), ("demo", "asset")]


def test_parquet(monkeypatch):
    buffer = BytesIO()
    pd.DataFrame({"x": [1, 2], "y": [0.5, 1.5]}).to_parquet(buffer)
    serve(monkeypatch, buffer.getvalue())
    assert retrieve_parquet.entrypoint("demo").to_dict("records") == [{"x": 1, "y": 0.5}, {"x": 2, "y": 1.5}]


def test_parquet_without_pyarrow(monkeypatch):
    serve(monkeypatch, b"")
    monkeypatch.setitem(__import__("sys").modules, "pyarrow", None)
    with pytest.raises(SDKError, match="pyarrow"):
        retrieve_parquet.entrypoint("demo")


def test_json(monkeypatch):
    serve(monkeypatch, b'{"a": [1, 2]}')
    assert retrieve_json.entrypoint("demo") == {"a": [1, 2]}


def test_jsonl(monkeypatch):
    serve(monkeypatch, b'{"x": 1, "y": "a"}\n{"x": 2, "y": "b"}\n')
    assert retrieve_jsonl.entrypoint("demo").to_dict("records") == [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]


@pytest.mark.parametrize(("step", "payload"), [(retrieve_json, b"{nope"), (retrieve_json, b"\xff"), (retrieve_jsonl, b"{nope")])
def test_malformed_input(monkeypatch, step, payload):
    serve(monkeypatch, payload)
    with pytest.raises(SDKError):
        step.entrypoint("demo")
