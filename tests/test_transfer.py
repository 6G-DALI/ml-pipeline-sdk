from concurrent.futures import ThreadPoolExecutor

import boto3
from moto import mock_aws
import pytest
import requests

from dali_mlops.catalogue import select_distribution
from dali_mlops.edc import EDC, TransferConfig, retrieve
from dali_mlops.storage import Staging
from dali_mlops.types import SDKError


def graph(*assets):
    return {"@graph": [
        {"@id": "https://catalog.example/datasets/demo", "dcat:distribution": [
            {"@id": f"urn:{asset}"} for asset in assets
        ]},
        *[{"@id": f"urn:{asset}", "https://dali-project.eu/ns#assetId": [{"@value": asset}],
           "dcat:accessURL": [{"@id": "https://provider.example"}],
           "dct:title": {"@value": f"{asset}.csv"}, "dcat:mediaType": "text/csv"}
          for asset in assets],
    ]}


def test_distribution_selection():
    with pytest.raises(SDKError, match="no fetchable"):
        select_distribution(graph(), "demo")
    assert select_distribution(graph("a"), "demo").asset_id == "a"
    assert select_distribution({"@graph": graph("a")["@graph"][1:]}, "demo").asset_id == "a"
    with pytest.raises(SDKError, match=r"a \(a.csv, text/csv\), b"):
        select_distribution(graph("a", "b"), "demo")
    assert select_distribution(graph("a", "b"), "demo", "b").asset_id == "b"
    with pytest.raises(SDKError, match="does not belong"):
        select_distribution(graph("a"), "demo", "b")


def response(status=200, payload=None, data=b""):
    import json
    from io import BytesIO

    result = requests.Response()
    result.status_code = status
    result.raw = BytesIO(json.dumps(payload).encode() if payload is not None else data)
    return result


@pytest.mark.parametrize(("mode", "missing"), [("push", False), ("pull", False), ("push", True)])
@mock_aws
def test_transfer(monkeypatch, mode, missing):
    from urllib.parse import urlsplit

    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="datasets")
    staging = Staging(client, "datasets", client)
    edr_requests = []
    payloads = []

    def fake(method, url, **kwargs):
        path = urlsplit(url).path
        if path == "/datasets/demo":
            assert kwargs["headers"] == {"Accept": "application/ld+json"}
            return response(payload=graph("a"))
        if path == "/download":
            assert "X-Api-Key" not in kwargs["headers"]
            return response(data=b"x,y\n")
        assert kwargs["headers"] == {"X-Api-Key": "credential"}
        payload = kwargs["json"]
        if path.endswith("/catalog/request"):
            assert payload["querySpec"]["filterExpression"][0]["operandRight"] == "a"
            return response(payload={"dcat:dataset": [{
                "@id": "a", "odrl:hasPolicy": {"@id": "offer", "odrl:permission": [{"odrl:action": "use"}]},
            }]})
        if path.endswith("/contractnegotiations"):
            assert payload["policy"]["odrl:target"] == {"@id": "a"}
            assert payload["policy"]["odrl:permission"] == [{"odrl:action": "use"}]
            return response(payload={"@id": "neg"})
        if path.endswith("/contractnegotiations/neg"):
            return response(payload={"edc:state": "FINALIZED", EDC + "contractAgreementId": "agreement"})
        if path.endswith("/transferprocesses"):
            payloads.append(payload)
            if mode == "push":
                destination = payload["dataDestination"]
                assert destination["method"] == "PUT"
                key = urlsplit(destination["baseUrl"]).path.lstrip("/")
                if not missing:
                    client.put_object(Bucket="datasets", Key=key, Body=b"x,y\n")
            else:
                assert "dataDestination" not in payload
            return response(payload={"@id": "transfer"})
        if path.endswith("/transferprocesses/transfer"):
            return response(payload={"state": "COMPLETED" if mode == "push" else "STARTED"})
        if path.endswith("/edrs/transfer/dataaddress"):
            edr_requests.append(1)
            if len(edr_requests) == 1:
                return response(status=404)
            return response(payload={EDC + "endpoint": [{"@value": "https://download.example/download?sig=secret"}]})
        raise AssertionError(path)

    monkeypatch.setattr(requests, "request", fake)
    config = TransferConfig("https://catalog.example", "https://consumer.example", None, "participant",
                            mode, interval=0)
    if missing:
        with pytest.raises(SDKError, match="missing or unreadable"):
            retrieve("demo", None, config, {"X-Api-Key": "credential"}, staging)
        return
    if mode == "push":
        with ThreadPoolExecutor(max_workers=2) as pool:
            files = list(pool.map(lambda _: retrieve("demo", None, config, {"X-Api-Key": "credential"}, staging),
                                  range(2)))
        assert files[0].storage_uri != files[1].storage_uri
    else:
        files = [retrieve("demo", None, config, {"X-Api-Key": "credential"}, staging)]
        assert len(edr_requests) == 2
    for result in files:
        assert (result.filename, result.media_type, result.agreement_id, result.transfer_id) == (
            "a.csv", "text/csv", "agreement", "transfer")
        assert client.get_object(Bucket="datasets", Key=urlsplit(result.storage_uri).path[1:])["Body"].read() == b"x,y\n"
        assert "secret" not in repr(result)


def test_wait_and_redaction(monkeypatch):
    from dali_mlops.edc import Consumer

    consumer = Consumer("https://consumer.example", {}, TransferConfig("", None, None, "", timeout=0.01, interval=0))
    monkeypatch.setattr(consumer, "call", lambda *args, **kwargs: {"state": "IN_PROGRESS"})
    with pytest.raises(SDKError, match="Timed out"):
        consumer.wait("/contractnegotiations/1", ("FINALIZED",))
    monkeypatch.setattr(consumer, "call", lambda *args, **kwargs: {"state": "TERMINATED", "errorDetail": "?sig=secret"})
    with pytest.raises(SDKError, match=r"^EDC reached terminal state TERMINATED[^?]+$"):
        consumer.wait("/contractnegotiations/1", ("FINALIZED",))
    monkeypatch.setattr(requests, "request", lambda *args, **kwargs: response(500, data=b"?sig=secret"))
    with pytest.raises(SDKError, match=r"^Piveau dataset lookup returned HTTP 500\.$"):
        retrieve("demo", None, consumer.config.__class__("https://catalog.example", None, None, "p"), {}, None)


@mock_aws
def test_missing_staging_object():
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="datasets")
    with pytest.raises(SDKError, match="missing or unreadable"):
        Staging(client, "datasets").verify("missing")


def test_stack_connection_signs_provider_endpoint(monkeypatch):
    from types import SimpleNamespace
    from urllib.parse import parse_qs, urlsplit

    from dali_mlops import storage

    store = SimpleNamespace(
        path="s3://artifacts",
        config=SimpleNamespace(client_kwargs={"endpoint_url": "http://internal.example"}),
        get_credentials=lambda: ("test-key", "test-secret", "session-token", "us-west-1"),
    )
    monkeypatch.setattr(storage, "active_artifact_store", lambda: store)
    monkeypatch.setenv("DALI_STAGING_BUCKET", "datasets")
    monkeypatch.setenv("DALI_STAGING_SIGN_ENDPOINT", "https://external.example")
    staging = storage.staging_connection(push=True)
    assert staging.client.meta.endpoint_url == "http://internal.example"
    signed_url = urlsplit(staging.presign("dali-staging/key/file.bin", 60))
    query = parse_qs(signed_url.query)
    assert (signed_url.netloc, signed_url.path) == ("external.example", "/datasets/dali-staging/key/file.bin")
    assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
    assert query["X-Amz-Security-Token"] == ["session-token"]
    assert "test-key" in query["X-Amz-Credential"][0]
