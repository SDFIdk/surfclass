# pylint: disable=redefined-outer-name
import os
import pytest
from click.testing import CliRunner


@pytest.fixture(scope="function")
def cli_runner():
    return CliRunner()


@pytest.fixture(scope="session")
def data_dir():
    """Absolute file path to the dir with test data."""
    return os.path.abspath(os.path.join("tests", "data"))


@pytest.fixture(scope="session")
def las_filepath(data_dir):
    return os.path.join(data_dir, "1km_6171_727_decimated.las")
