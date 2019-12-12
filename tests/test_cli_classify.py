from pathlib import Path
from osgeo import gdal
import numpy as np
from surfclass.scripts.cli import cli


def test_cli_classify(cli_runner):
    result = cli_runner.invoke(cli, ["classify"], catch_exceptions=False)
    assert result.exit_code == 0


def test_cli_classify_genericmodel(cli_runner, genericmodel_filepath, tmp_path):
    classification_data_path = (
        Path("./tests/data/classification_data").absolute().resolve()
    )
    f1 = classification_data_path.joinpath("6171_727_amplitude.tif")
    f2 = classification_data_path.joinpath("6171_727_diffmean_n3.tif")
    f3 = classification_data_path.joinpath("6171_727_mean_n3.tif")
    f4 = classification_data_path.joinpath("6171_727_var_n3.tif")

    # f1 ... f4 are mapped to the multiple argument -f
    args = (
        f"classify genericmodel -b 727000 6171000 728000 6172000 -f {f1} "
        f"-f {f2} -f {f3} -f {f4} {genericmodel_filepath} {tmp_path}/classification.tif"
    )
    result = cli_runner.invoke(cli, args.split(" "), catch_exceptions=False)
    assert result.exit_code == 0

    outfile = tmp_path / "classification.tif"
    ds = gdal.Open(str(outfile))
    srcband = ds.GetRasterBand(1)
    nodata = srcband.GetNoDataValue()
    prediction = ds.ReadAsArray()
    assert prediction.dtype == "uint8"
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


def test_cli_classify_genericmodel_prob(cli_runner, genericmodel_filepath, tmp_path):
    # Use the same model from genericmodel but using the generic classification command
    classification_data_path = (
        Path("./tests/data/classification_data").absolute().resolve()
    )
    f1 = classification_data_path.joinpath("6171_727_amplitude.tif")
    f2 = classification_data_path.joinpath("6171_727_diffmean_n3.tif")
    f3 = classification_data_path.joinpath("6171_727_mean_n3.tif")
    f4 = classification_data_path.joinpath("6171_727_var_n3.tif")

    # f1 ... f4 are mapped to the multiple argument -f
    args = (
        f"classify genericmodel -b 727000 6171000 728000 6172000 -f {f1} "
        f"-f {f2} -f {f3} -f {f4} {genericmodel_filepath} --prob {tmp_path}/classification_prob.tif {tmp_path}/classification.tif"
    )
    result = cli_runner.invoke(cli, args.split(" "), catch_exceptions=False)
    assert result.exit_code == 0

    outfile = tmp_path / "classification_prob.tif"
    ds = gdal.Open(str(outfile))
    srcband = ds.GetRasterBand(1)
    nodata = srcband.GetNoDataValue()
    prediction_prob = ds.ReadAsArray()
    assert prediction_prob.dtype == "float32"
    assert ds.GetGeoTransform() == (727000, 4, 0, 6172000, 0, -4)
    assert prediction_prob.shape == (250, 250)

    # Upper Left corner has 0.86 probability
    assert np.isclose(prediction_prob[0, 0], 0.86)

    # This is a hole in the mask, and should always be nodata which is 0
    assert prediction_prob[3, 2] == int(nodata) == 0
