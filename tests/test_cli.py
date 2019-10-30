from click.testing import CliRunner
import surfclass
from surfclass.scripts.cli import cli


def test_cli():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])

    expected = f"surfclass, version {surfclass.__version__}\n"
    assert result.output == expected
