import pytest
import copy
import sys
import os

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


@pytest.fixture(autouse=True)
def clear_settings():
    yield
    from elysia.config import DEFAULT_SETTINGS, settings

    settings.inherit_config(copy.deepcopy(DEFAULT_SETTINGS))
