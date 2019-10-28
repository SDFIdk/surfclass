import os
import pytest
from click.testing import CliRunner
from surfclass.scripts.cli import cli
import surfclass


def test_cli():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0


def test_gdal():
    from osgeo import gdal

    assert gdal.VersionInfo() == ""

