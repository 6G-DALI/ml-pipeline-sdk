from io import BytesIO

import pytest

from dali_mlops import DatasetFile, retrieve_csv, retrieve_dataset, retrieve_json, retrieve_jsonl, retrieve_parquet, retrieve_tsv
from dali_mlops import retrieval


@pytest.mark.parametrize("stub", [retrieve_parquet, retrieve_json, retrieve_jsonl])
def test_stubs_precede_io(monkeypatch, stub):
    monkeypatch.setattr(retrieval, "transfer_config", lambda: pytest.fail("configuration lookup"))
    with pytest.raises(NotImplementedError, match=r'retrieve_dataset\(\).*DatasetFile.open\("rb"\)'):
        stub.entrypoint("demo")


@pytest.mark.parametrize(("reader", "delimiter"), [(retrieve_csv, b","), (retrieve_tsv, b"\t")])
def test_readers(monkeypatch, reader, delimiter):
    reference = DatasetFile("file", None, "demo", "asset", "s3://datasets/object", "agreement", "transfer", 8)
    called = []

    def fetch(dataset, asset):
        called.append((dataset, asset))
        return reference

    monkeypatch.setattr(retrieval, "fetch_dataset", fetch)
    monkeypatch.setattr(DatasetFile, "open", lambda self, mode: BytesIO(delimiter.join([b"x", b"y\n1", b"2\n"])))
    assert reader.entrypoint("demo", "asset").to_dict("records") == [{"x": 1, "y": 2}]
    assert retrieve_dataset.entrypoint("demo", "asset") is reference
    assert called == [("demo", "asset"), ("demo", "asset")]
