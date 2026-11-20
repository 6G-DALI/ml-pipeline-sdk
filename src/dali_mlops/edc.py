"""Consumer management API and Data Space transfer state machine."""

from dataclasses import dataclass
import re
from time import monotonic, sleep
from urllib.parse import quote

from .catalogue import check_status, discover, field, http_url, identifier, items, json_response, request, scalar
from .types import DatasetFile, SDKError

EDC = "https://w3id.org/edc/v0.0.1/ns/"
ODRL = "http://www.w3.org/ns/odrl/2/"
DCAT = "http://www.w3.org/ns/dcat#"
PROTOCOL = "dataspace-protocol-http"
TERMINAL = {"TERMINATED", "TERMINATING", "ERROR", "FAILED", "CANCELLED", "DEPROVISIONED", "COMPLETED"}


def property_value(node, name):
    return scalar(field(node, name, f"edc:{name}", EDC + name))


@dataclass(frozen=True)
class TransferConfig:
    catalogue: str
    consumer: str | None
    provider: str | None
    provider_id: str
    mode: str = "push"
    timeout: float = 120
    interval: float = 1
    request_timeout: float = 30
    presign_seconds: int = 900
    download_timeout: float = 300


class Consumer:
    def __init__(self, base, headers, config):
        self.base = http_url(base).rstrip("/") + "/management/v3"
        self.headers = headers
        self.config = config

    def call(self, method, path, payload=None, *, timeout=None, pending=False):
        operation = "EDC " + path.strip("/").split("/")[0]
        response = request(method, self.base + path, operation, json=payload,
                           headers=self.headers, timeout=timeout or self.config.request_timeout)
        if pending and response.status_code in (404, 204):
            response.close()
            return {}
        return json_response(response, operation)

    def poll(self, operation, probe):
        deadline = monotonic() + self.config.timeout
        while (remaining := deadline - monotonic()) > 0:
            result = probe(min(remaining, self.config.request_timeout))
            if result:
                return result
            sleep(min(self.config.interval, max(0, deadline - monotonic())))
        raise SDKError(f"Timed out waiting for {operation}.")

    def wait(self, path, expected):
        def probe(timeout):
            document = self.call("GET", path, timeout=timeout)
            state = property_value(document, "state")
            if state in expected:
                return document
            if state in TERMINAL:
                raise SDKError(f"EDC reached terminal state {state} while waiting for {' or '.join(expected)}.")
            return None

        return self.poll("EDC state " + " or ".join(expected), probe)


def retrieve(dataset_id, asset_id, config, headers, staging):
    distribution = discover(config.catalogue, dataset_id, asset_id, config.request_timeout)
    asset = distribution.asset_id
    provider = http_url(config.provider or distribution.connector).rstrip("/")
    dsp = provider if provider.endswith("/protocol") else provider + "/protocol"
    consumer = Consumer(config.consumer or provider.removesuffix("/protocol"), headers, config)
    context = {"@vocab": EDC, "odrl": ODRL}
    common = {"@context": context, "counterPartyAddress": dsp, "protocol": PROTOCOL}
    catalogue = consumer.call("POST", "/catalog/request", {
        **common,
        "querySpec": {"filterExpression": [{"operandLeft": EDC + "id", "operator": "=", "operandRight": asset}]},
    })
    entries = items(field(catalogue, "dcat:dataset", DCAT + "dataset", "dataset"))
    entry = next((item for item in entries if isinstance(item, dict) and
                  (scalar(item.get("@id")) or property_value(item, "id")) == asset), None)
    if entry is None:
        raise SDKError("The provider returned no catalogue entry for the selected asset.")
    offers = [offer for offer in items(field(entry, "odrl:hasPolicy", ODRL + "hasPolicy", "hasPolicy"))
              if isinstance(offer, dict) and scalar(offer.get("@id"))]
    if not offers:
        raise SDKError("The selected provider asset has no policy offer.")
    policy = dict(offers[0])
    target = scalar(field(policy, "odrl:target", ODRL + "target", "target"))
    if target and target != asset:
        raise SDKError("The provider offer target does not match the selected asset.")
    if not target:
        policy["odrl:target"] = {"@id": asset}
    if field(policy, "odrl:assigner", ODRL + "assigner", "assigner") is None:
        policy["odrl:assigner"] = {"@id": config.provider_id}
    policy.setdefault("@type", ODRL + "Offer")
    negotiation = consumer.call("POST", "/contractnegotiations", {
        **common, "@type": "ContractRequest", "providerId": config.provider_id, "policy": policy,
    })
    negotiation_id = quote(identifier(negotiation.get("@id")), safe="")
    finalized = consumer.wait(f"/contractnegotiations/{negotiation_id}", ("FINALIZED",))
    agreement = identifier(property_value(finalized, "contractAgreementId"))
    media = distribution.media_type or property_value(entry, "contenttype")
    if media:
        media = media.partition(";")[0].strip().lower()
        if not re.fullmatch(r"[\w.+-]+/[\w.+-]+", media):
            media = None
    name = distribution.filename(property_value(entry, "name"), media)
    key = staging.new_key(name)
    transfer_request = {
        **common, "@type": "TransferRequest", "connectorId": config.provider_id,
        "contractId": agreement, "assetId": asset,
    }
    if config.mode == "push":
        staging.ensure_push_signer()
        transfer_request.update(transferType="PresignedHttpData-PUSH", dataDestination={
            "type": "PresignedHttpData", "baseUrl": staging.presign(key, config.presign_seconds),
            "method": "PUT",
        })
    elif config.mode == "pull":
        transfer_request["transferType"] = "PresignedUrl-PULL"
    else:
        raise SDKError("Set DALI_TRANSFER_MODE to push or pull.")
    transfer = consumer.call("POST", "/transferprocesses", transfer_request)
    transfer_id = identifier(transfer.get("@id"))
    path_id = quote(transfer_id, safe="")
    if config.mode == "push":
        consumer.wait(f"/transferprocesses/{path_id}", ("COMPLETED",))
    else:
        consumer.wait(f"/transferprocesses/{path_id}", ("STARTED", "COMPLETED"))

        def edr(timeout):
            address = consumer.call("GET", f"/edrs/{path_id}/dataaddress", timeout=timeout, pending=True)
            return property_value(address, "endpoint") or property_value(address, "baseUrl")

        signed_url = consumer.poll("the signed endpoint in the transfer EDR", edr)
        with request("GET", http_url(signed_url, signed=True), "Signed dataset download",
                     stream=True, timeout=config.request_timeout, headers={"Accept-Encoding": "identity"}) as download:
            check_status(download, "Signed dataset download")
            staging.stage_download(key, download, config.download_timeout)
    size = staging.verify(key)
    return DatasetFile(name, media, dataset_id, asset, staging.uri(key), agreement, transfer_id, size)
