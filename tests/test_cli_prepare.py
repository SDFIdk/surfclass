from surfclass.scripts.cli import cli


def test_cli_prepare(cli_runner):
    result = cli_runner.invoke(cli, ["prepare"])
    assert result.exit_code == 0


def test_cli_prepare_lidar(cli_runner):
    result = cli_runner.invoke(cli, ["prepare", "lidar"])
    assert result.exit_code == 0
