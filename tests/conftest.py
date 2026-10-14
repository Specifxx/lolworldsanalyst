import pytest

from lolworlds.config import load_settings


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture
def aggressive(settings):
    return settings.risk_profiles["aggressive"]
