import pytest

from dali_mlops import config, runtime

KEYS = list(config.DEFAULTS)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config, "optional_secret", lambda name: {})


def test_defaults():
    assert config.setting("PIVEAU_HUB_URL") == "https://dataspace.6gdali.eu"
    assert config.setting("EDC_PROVIDER_URL") == config.setting("EDC_CONSUMER_URL") == "https://edc.dataspace.6gdali.eu"
    assert config.setting("EDC_PROVIDER_ID") == "provider-facility"
    assert config.datalake_bucket() == "6g-dali-models"


def test_secret_overrides_default(monkeypatch):
    secrets = {"dataspace": {"EDC_PROVIDER_ID": "from-secret"}, "datalake-model-storage": {"DLAKE_BUCKET": "other"}}
    monkeypatch.setattr(config, "optional_secret", lambda name: secrets[name])
    assert config.setting("EDC_PROVIDER_ID") == "from-secret"
    assert config.setting("PIVEAU_HUB_URL") == "https://dataspace.6gdali.eu"
    assert config.datalake_bucket() == "other"


def test_environment_overrides_secret(monkeypatch):
    monkeypatch.setattr(config, "optional_secret", lambda name: {"EDC_PROVIDER_ID": "from-secret"})
    monkeypatch.setenv("EDC_PROVIDER_ID", "from-env")
    assert config.setting("EDC_PROVIDER_ID") == "from-env"


def test_secret_is_not_read_when_the_environment_answers(monkeypatch):
    monkeypatch.setattr(config, "optional_secret", lambda name: pytest.fail("secret read"))
    monkeypatch.setenv("PIVEAU_HUB_URL", "https://hub.example")
    assert config.setting("PIVEAU_HUB_URL") == "https://hub.example"


def test_unreadable_secret_falls_back(monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr("zenml.client.Client.get_secret", lambda self, name: (_ for _ in ()).throw(RuntimeError))
    assert config.optional_secret("dataspace") == {}


def test_transfer_config_needs_no_environment(monkeypatch):
    cfg = runtime.transfer_config()
    assert (cfg.catalogue, cfg.consumer, cfg.provider, cfg.provider_id) == (
        "https://dataspace.6gdali.eu", "https://edc.dataspace.6gdali.eu",
        "https://edc.dataspace.6gdali.eu", "provider-facility")
    monkeypatch.setenv("EDC_PROVIDER_ID", "other-provider")
    assert runtime.transfer_config().provider_id == "other-provider"
