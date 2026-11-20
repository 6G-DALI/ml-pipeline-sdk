"""Resolve worker configuration only when a step runs."""

import math
import os

from .catalogue import identifier
from .edc import TransferConfig
from .types import SDKError


def secret_values(name, *required):
    from zenml.client import Client

    try:
        values = Client().get_secret(name).secret_values
    except Exception:
        raise SDKError(f"Could not read ZenML secret '{name}'. Check the active stack and secret access.") from None
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise SDKError(f"ZenML secret '{name}' is missing fields: {', '.join(missing)}.")
    return values


def positive_env(key, default, convert=float):
    try:
        value = convert(os.environ.get(key, default))
        if math.isfinite(value) and value > 0:
            return value
    except (ValueError, OverflowError):
        pass
    raise SDKError(f"Set {key} to a positive number.")


def transfer_config():
    env = os.environ
    for key in ("PIVEAU_HUB_URL", "EDC_PROVIDER_ID"):
        if not env.get(key):
            raise SDKError(f"Set {key} in the worker environment.")
    mode = env.get("DALI_TRANSFER_MODE", "push").lower()
    if mode not in ("push", "pull"):
        raise SDKError("Set DALI_TRANSFER_MODE to push or pull.")
    return TransferConfig(
        catalogue=env["PIVEAU_HUB_URL"],
        consumer=env.get("EDC_CONSUMER_URL"),
        provider=env.get("EDC_PROVIDER_URL"),
        provider_id=identifier(env["EDC_PROVIDER_ID"]),
        mode=mode,
        timeout=positive_env("DALI_POLL_TIMEOUT", 120),
        interval=positive_env("DALI_POLL_INTERVAL", 1),
        request_timeout=positive_env("DALI_REQUEST_TIMEOUT", 30),
        presign_seconds=positive_env("DALI_PRESIGN_SECONDS", 900, int),
        download_timeout=positive_env("DALI_DOWNLOAD_TIMEOUT", 300),
    )
