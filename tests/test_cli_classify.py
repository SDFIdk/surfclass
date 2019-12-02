from pathlib import Path
from osgeo import gdal
from surfclass.scripts.cli import cli
from surfclass.randomforestclassifier import RandomForestClassifier


def test_cli_classify(cli_runner):
    result = cli_runner.invoke(cli, ["classify"], catch_exceptions=False)
    assert result.exit_code == 0


def test_cli_classify_testmodel1(cli_runner, testmodel1_filepath, data_dir, tmp_path):

    classification_data_path = (
        Path("./tests/data/classification_data").absolute().resolve()
    )
    f1 = classification_data_path.joinpath("6171_727_amplitude.tif")
    f2 = classification_data_path.joinpath("6171_727_mean_n3.tif")
    f3 = classification_data_path.joinpath("6171_727_var_n3.tif")
    f4 = classification_data_path.joinpath("6171_727_diffmean_n3.tif")

    args = f"classify testmodel1 -b 727000 6171000 728000 6172000 -f1 {f1} -f2 {f2} -f3 {f3} -f4 {f4} {testmodel1_filepath} {data_dir}"

    result = cli_runner.invoke(cli, args.split(" "), catch_exceptions=False)
    assert result.exit_code == 0

    outfile = data_dir / "prediction.tif"
    ds = gdal.Open(str(outfile))
    assert ds.GetGeoTransform() == (727000, 4, 0, 6172000, 0, -4)

    # TODO: Test that the actual classification gives expected result

