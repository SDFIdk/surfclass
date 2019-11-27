# pylint: disable=redefined-outer-name
from pathlib import Path
import pytest
from click.testing import CliRunner


@pytest.fixture(scope="function")
def cli_runner():
    return CliRunner()


@pytest.fixture(scope="session")
def data_dir():
    """Absolute file path to the dir with test data."""
    return Path("./tests/data").absolute().resolve()


@pytest.fixture(scope="session")
def las_filepath(data_dir):
    return Path(data_dir) / "1km_6171_727_decimated.las"


@pytest.fixture(scope="session")
def classraster_filepath(data_dir):
    return Path(data_dir) / "classes.tif"


@pytest.fixture(scope="session")
def polygons_filepath(data_dir):
    return Path(data_dir) / "polygons.geojson"


@pytest.fixture(scope="session")
def amplituderaster_filepath(data_dir):
    return Path(data_dir) / "1km_6171_727_amplitude_4m.tif"
