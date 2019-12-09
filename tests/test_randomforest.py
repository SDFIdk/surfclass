from surfclass.randomforest import RandomForest
from surfclass.train import (
    collect_training_data,
    save_training_data,
    load_training_data,
)


def test_randomforest_train(polygons_filepath, data_dir, tmp_path):
    rasters = [
        "6171_727_amplitude.tif",
        "6171_727_diffmean_n3.tif",
        "6171_727_mean_n3.tif",
        "6171_727_var_n3.tif",
    ]
    # Setup the test by creating some training data first
    rasters = [data_dir / "classification_data" / x for x in rasters]
    classes, features = collect_training_data(polygons_filepath, None, "id", rasters)

    outfile = tmp_path / "test.npz"
    save_training_data(outfile, rasters, classes, features)

    assert outfile.exists()

    # The CLI entry point will point to file with training data
    _, read_classes, read_features = load_training_data(outfile)

    # Extract some parameters based on data
    num_features = read_features.shape[1]

    # Define the RandomForest model with the training data loaded
    # Use a low amount of trees to reduce test time
    model = RandomForest(num_features, 10, model=None)

    trained_model = model.train(read_features, read_classes)
    assert trained_model.n_features_ == num_features
    assert trained_model.n_estimators == 10
