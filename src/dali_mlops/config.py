"""Worker settings: step environment, then ZenML secret, then the 6G-DALI production defaults."""

import os

DEFAULTS = {
    "PIVEAU_HUB_URL": "https://dataspace.6gdali.eu",
    "EDC_PROVIDER_URL": "https://edc.dataspace.6gdali.eu",
    "EDC_CONSUMER_URL": "https://edc.dataspace.6gdali.eu",
    "EDC_PROVIDER_ID": "provider-facility",
    "DLAKE_BUCKET": "6g-dali-models",
}
# Keys that may be overridden in the ZenML secret of the same group. Keys themselves live only in secrets.
SECRET_GROUPS = {
    "dataspace": ("PIVEAU_HUB_URL", "EDC_PROVIDER_URL", "EDC_CONSUMER_URL", "EDC_PROVIDER_ID"),
    "datalake-model-storage": ("DLAKE_BUCKET",),
}
_GROUP_OF = {key: group for group, keys in SECRET_GROUPS.items() for key in keys}


def optional_secret(name):
    """Return the secret's values, or an empty mapping when it is missing or unreadable."""
    from zenml.client import Client

    try:
        return dict(Client().get_secret(name).secret_values)
    except Exception:
        return {}


def setting(key):
    """Resolve one setting: environment, then secret, then the SDK default."""
    if value := os.environ.get(key):
        return value
    if group := _GROUP_OF.get(key):
        if value := optional_secret(group).get(key):
            return value
    return DEFAULTS[key]


def datalake_bucket():
    return setting("DLAKE_BUCKET")
