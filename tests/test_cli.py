import surfclass
from surfclass.scripts.cli import cli


def test_cli(cli_runner):
    result = cli_runner.invoke(cli, ["--version"], catch_exceptions=False)
    assert result.exit_code == 0


def test_cli_version(cli_runner):
    result = cli_runner.invoke(cli, ["--version"], catch_exceptions=False)
    expected = f"surfclass, version {surfclass.__version__}\n"
    assert result.output == expected
