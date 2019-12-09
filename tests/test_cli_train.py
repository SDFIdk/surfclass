from osgeo import gdal
from surfclass.scripts.cli import cli


def test_cli_train(cli_runner):
    result = cli_runner.invoke(cli, ["train"], catch_exceptions=False)
    assert result.exit_code == 0
