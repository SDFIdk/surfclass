import pytest
from click.testing import CliRunner


@pytest.fixture(scope="function")
def cli_runner():
    return CliRunner()
