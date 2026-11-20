"""Resolve connector-backed distributions from Piveau JSON-LD."""

from dataclasses import dataclass
from urllib.parse import quote, urlsplit

import requests

from .types import SDKError, safe_filename

DALI = "https://dali-project.eu/ns#"
DCAT = "http://www.w3.org/ns/dcat#"
DCT = "http://purl.org/dc/terms/"


def items(value):
    return value if isinstance(value, list) else ([] if value is None else [value])


def scalar(value):
    for item in items(value):
        if isinstance(item, dict):
            item = item.get("@id") or item.get("@value")
        if isinstance(item, str) and item.strip():
            return item.strip()
    return None


def field(node, *keys):
    return next((node[key] for key in keys if key in node), None)


def identifier(value):
    value = scalar(value)
    if not value or len(value) > 512 or not all(c.isalnum() or c in "-_.:" for c in value):
        raise SDKError("Expected a plain Data Space identifier.")
    return value


def http_url(value, *, signed=False):
    try:
        parsed = urlsplit(value or "")
        if (parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username
                and not parsed.password and not parsed.fragment and (signed or not parsed.query)):
            return value
    except ValueError:
        pass
    raise SDKError("Expected a valid HTTP endpoint.")


def request(method, url, operation, **kwargs):
    try:
        response = requests.request(method, url, allow_redirects=False, **kwargs)
    except (requests.RequestException, ValueError):
        raise SDKError(f"{operation} could not connect or timed out.") from None
    return response


def check_status(response, operation):
    if not 200 <= response.status_code < 300:
        response.close()
        raise SDKError(f"{operation} returned HTTP {response.status_code}.")


def json_response(response, operation):
    with response:
        check_status(response, operation)
        try:
            result = response.json()
        except ValueError:
            raise SDKError(f"{operation} returned invalid JSON.") from None
    if not isinstance(result, dict):
        raise SDKError(f"{operation} returned an unexpected JSON document.")
    return result


@dataclass(frozen=True)
class Distribution:
    asset_id: str
    connector: str
    title: str | None
    media_type: str | None

    def filename(self, offer_name=None, media_type=None):
        for candidate in (offer_name, self.title):
            if candidate and "." in candidate:
                try:
                    return safe_filename(candidate)
                except ValueError:
                    pass
        suffix = {
            "text/csv": ".csv", "text/tab-separated-values": ".tsv",
            "application/json": ".json", "application/x-ndjson": ".jsonl",
            "application/vnd.apache.parquet": ".parquet",
        }.get(media_type or self.media_type, ".bin")
        return "dataset" + suffix


def select_distribution(document, dataset_id, asset_id=None):
    nodes = [node for node in items(document.get("@graph", document)) if isinstance(node, dict)]
    by_id = {node["@id"]: node for node in nodes if isinstance(node.get("@id"), str)}
    datasets = [
        node for node in nodes
        if field(node, DCAT + "distribution", "dcat:distribution") is not None
    ]
    matching = [
        node for node in datasets
        if (scalar(node.get("@id")) or "").rstrip("/").rsplit("/", 1)[-1] == dataset_id
    ]
    if not matching and len(datasets) == 1:
        matching = datasets
    references = [
        ref for node in matching
        for ref in items(field(node, DCAT + "distribution", "dcat:distribution"))
    ]
    if not datasets:
        references = [
            node for node in nodes
            if any(scalar(kind) in (DCAT + "Distribution", "dcat:Distribution")
                   for kind in items(node.get("@type")))
        ]
    distributions = {}
    for ref in references:
        node = by_id.get(scalar(ref), ref)
        if not isinstance(node, dict):
            continue
        asset = scalar(field(node, DALI + "assetId", "dali:assetId"))
        access = scalar(field(node, DCAT + "accessURL", "dcat:accessURL"))
        if not asset or not access:
            continue
        asset = identifier(asset)
        title = scalar(field(node, DCT + "title", "dct:title"))
        media = scalar(field(node, DCAT + "mediaType", "dcat:mediaType", DCT + "format", "dct:format"))
        if media and media.startswith("https://www.iana.org/assignments/media-types/"):
            media = media.split("/media-types/", 1)[1]
        distributions[asset] = Distribution(asset, http_url(access), title, media)
    if not distributions:
        raise SDKError("The dataset has no fetchable connector-backed distributions.")
    if asset_id is not None:
        if identifier(asset_id) not in distributions:
            raise SDKError("The requested asset does not belong to this dataset's fetchable distributions.")
        return distributions[asset_id]
    if len(distributions) != 1:
        # Catalogue labels are untrusted. Exclude URL-like strings and control characters.
        def label(value):
            return value[:120] if value and all(c.isalnum() or c in " ._-/()" for c in value) else "unknown"

        choices = ", ".join(
            f"{d.asset_id} ({label(d.title)}, {label(d.media_type)})"
            for d in distributions.values()
        )
        raise SDKError(f"Select asset_id from the fetchable distributions: {choices}.")
    return next(iter(distributions.values()))


def discover(base, dataset_id, asset_id, timeout):
    url = f"{http_url(base).rstrip('/')}/datasets/{quote(identifier(dataset_id), safe='')}"
    operation = "Piveau dataset lookup"
    response = request("GET", url, operation, headers={"Accept": "application/ld+json"}, timeout=timeout)
    return select_distribution(json_response(response, operation), dataset_id, asset_id)
