import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: full 3D runs (tens of seconds); -m 'not slow' to skip")
