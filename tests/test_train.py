import numpy as np
from surfclass import train


def test_collect_train_data(polygons_filepath, data_dir):
    # pylint: disable=E1136
    # Disable false classes.shape[0] is unsubscriptable
    rasters = [
        "6171_727_amplitude.tif",
        "6171_727_diffmean_n3.tif",
        "6171_727_mean_n3.tif",
    ]
    # Go absolute path
    rasters = [data_dir / "classification_data" / x for x in rasters]

    # We use the polygon "id" attribute as class
    classes, features = train.collect_training_data(
        polygons_filepath, None, "id", rasters
    )
    assert int(np.sum(classes)) == 55266
    assert features.shape == (classes.shape[0], 3)
    assert int(np.sum(features)) == 55961


def test_save_and_load(polygons_filepath, data_dir, tmp_path):
    # pylint: disable=E1136
    # Disable false classes.shape[0] is unsubscriptable
    # Create data the hard way...
    rasters = [
        "6171_727_amplitude.tif",
        "6171_727_diffmean_n3.tif",
        "6171_727_mean_n3.tif",
    ]
    rasters = [data_dir / "classification_data" / x for x in rasters]
    classes, features = train.collect_training_data(
        polygons_filepath, None, "id", rasters
    )

    outfile = tmp_path / "test.npz"
    train.save_training_data(outfile, rasters, classes, features)

    assert outfile.exists()

    read_files, read_classes, read_features = train.load_training_data(outfile)

    assert all([str(x[0]) == x[1] for x in zip(rasters, read_files)])
    np.testing.assert_equal(classes, read_classes)
    np.testing.assert_equal(features, read_features)
