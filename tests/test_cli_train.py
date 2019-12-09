from osgeo import gdal
from surfclass.scripts.cli import cli


def test_cli_train(cli_runner):
    result = cli_runner.invoke(cli, ["train"], catch_exceptions=False)
    assert result.exit_code == 0


def test_cli_train_testmodel1(cli_runner, tmp_path, testmodel1_traindata_filepath):
    args = f"train testmodel1 {testmodel1_traindata_filepath} {tmp_path}/tmp_model.sav"
    result = cli_runner.invoke(cli, args.split(" "), catch_exceptions=False)
    assert result.exit_code == 0

    # Check the file exists

    # Try calling predict using the model with the same data (use the testmodel1_traindata)

    # sanity checks
