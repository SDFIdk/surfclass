from pathlib import Path
from osgeo import gdal
import numpy as np
from surfclass.scripts.cli import cli


def test_cli_classify(cli_runner):
    result = cli_runner.invoke(cli, ["classify"], catch_exceptions=False)
    assert result.exit_code == 0


def test_cli_classify_testmodel1(cli_runner, testmodel1_filepath, tmp_path):

    classification_data_path = (
        Path("./tests/data/classification_data").absolute().resolve()
    )
    f1 = classification_data_path.joinpath("6171_727_amplitude.tif")
    f2 = classification_data_path.joinpath("6171_727_diffmean_n3.tif")
    f3 = classification_data_path.joinpath("6171_727_mean_n3.tif")
    f4 = classification_data_path.joinpath("6171_727_var_n3.tif")

    args = f"classify testmodel1 -b 727000 6171000 728000 6172000 -f1 {f1} -f2 {f2} -f3 {f3} -f4 {f4} {testmodel1_filepath} {tmp_path}"
    result = cli_runner.invoke(cli, args.split(" "), catch_exceptions=False)
    assert result.exit_code == 0

    outfile = tmp_path / "classification.tif"
    ds = gdal.Open(str(outfile))
    srcband = ds.GetRasterBand(1)
    nodata = srcband.GetNoDataValue()
    prediction = ds.ReadAsArray()
    assert ds.GetGeoTransform() == (727000, 4, 0, 6172000, 0, -4)
    assert prediction.shape == (250, 250)

    # Upper Left corner is grass
    assert prediction[0, 0] == 4
    # This is a hole in the mask, and should always be nodata which is 0
    assert prediction[3, 2] == int(nodata) == 0
    # Assert number of classes
    unique_elements, counts_elements = np.unique(prediction, return_counts=True)
    assert all(unique_elements == [0, 1, 2, 3, 4, 5])
    # Assert the counts of each class
    assert all(counts_elements == [4865, 22186, 80, 4863, 30243, 263])
