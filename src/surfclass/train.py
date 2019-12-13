"""Tools for training models."""
import logging
import numpy as np
from surfclass import rasterio, vectorize

logger = logging.getLogger(__name__)


def collect_training_data(poly_dataset, poly_layer, class_attribute, raster_paths):
    """Extracts training data defined by polygons with a class from a set of raster features.

    Training data consists of one or more polygons each defining an area of the same class. This method then extracts all
    cells within this area from the raster features and associates it with the polygon class.

    Args:
        poly_dataset (str or osgeo.ogr.DataSource): An OGR datasource. Either defined by its datasource string or a
                `osgeo.ogr.DataSource` object.
        poly_layer (str or osgeo.ogr.Layer, optional): Layer within the `datasource` to read from. Either a layer name,
                an `osgeo.ogr.Layer` object or `None`. If `None` the first layer in datasource is used. Defaults to None.
        class_attribute (str): Name of layer attribute which contains the class number for the polygon.
        raster_paths (list of str): List of paths to feature rasters. The order is important as the extracted feature data
                is stored in this order.

    Returns:
        tuple (ndarray, ndarray): A tuple with (classes, features) where classes.shape == (numobs,)
        and features.shape == (numobs, numfeatures).

    """
    # pylint: disable=E1136
    # Disable false classes.shape[0] is unsubscriptable
    logger.debug(
        "Collecting training data. Vector ds: '%s', Layer: '%s', Class attribute: '%s', Rasters: '%s'",
        poly_dataset,
        poly_layer,
        class_attribute,
        raster_paths,
    )
    f_paths = [str(p) for p in raster_paths]
    raster_readers = [rasterio.MaskedRasterReader(x) for x in f_paths]
    bbox = raster_readers[0].bbox
    gt = np.array(raster_readers[0].geotransform)
    shape = raster_readers[0].shape

    # Assert input raster are similar (same geotransform, same size)
    assert all(
        [np.allclose(r.geotransform, gt) for r in raster_readers]
    ), "All feature rasters must have same geotransform"
    assert all(
        r.shape == shape for r in raster_readers
    ), "All feature rasters must have same size"

    # Clip train features to raster extent
    featurereader = vectorize.FeatureReader(poly_dataset, poly_layer)
    featurereader.set_bbox_filter(bbox, clip=True)

    # List of arrays
    result_train = []

    # list of list of arrays
    # result[0] contains list of arrays from feature1
    # result[1] contains list of arrays from feature2
    result_features = [list() for x in range(len(f_paths))]

    for ogr_feature in featurereader:
        # What class does this poly represent?
        class_value = ogr_feature[class_attribute]
        # Get array of masked arrays containing features inside this poly
        cell_values = [r.read_flattened(ogr_feature.geometry()) for r in raster_readers]
        assert all([len(a) == len(cell_values[0]) for a in cell_values])
        # Get kombined mask of all features
        mask = np.zeros(cell_values[0].shape)
        for a in cell_values:
            mask = np.logical_or(mask, a.mask)
        # Get only valid (unmasked cells) from all arrays
        valid_mask = np.invert(mask)
        valid_cell_values = [a.data[valid_mask] for a in cell_values]
        # Create array of class_values matching length of feature arrays
        class_array = np.ones(valid_cell_values[0].shape) * class_value

        # Append to result
        result_train.append(class_array)
        for i, a in enumerate(valid_cell_values):
            result_features[i].append(a)

    # Now we have lists of matching arrays. Concatenate
    classes = np.concatenate(result_train)
    concat_features = [np.concatenate(a) for a in result_features]

    # stack and transpose into (num_cells, num_features) shape
    features = np.vstack(concat_features).T
    assert (
        features.shape[0] == classes.shape[0]
    ), "Classes and features do not have the same number of observations"
    assert features.shape[1] == len(
        f_paths
    ), "Features array does not have an entry per input feature"
    logger.debug("Collected training data: %s, %s", classes, features)
    return (classes, features)


def save_training_data(output_file, file_paths, classes, features):
    """Save training data to a file.

    Note:
        Order of `file_paths` MUST match the order of feature observations in `features`.

    Args:
        output_file (str or pathlib.Path): Path to output file.
        file_paths (list of str): List of paths to feature rasters. The order is important as this defines how the extracted
                feature data is stored.
        classes (ndarray): ndarray with class observations. Shape is (numobs, )
        features (ndarray): 2D ndarray with feature observations. Shape is (numobs, numfeatures).

    """
    assert output_file, "No output file specified"
    assert (
        classes.shape[0] == features.shape[0]
    ), "Number of class observations does not match number of feature observations."
    assert (
        len(file_paths) == features.shape[1]
    ), "Number of files does not match number of features."
    np.savez_compressed(
        output_file,
        file_paths=[str(x) for x in file_paths],
        classes=classes,
        features=features,
    )


def load_training_data(file_path):
    """Load training data from file as saved by `save_training_data`.

    Args:
        file_path (str or pathlib.Path): Path to file with saved data.

    Returns:
        tuple: A (file_paths, classes, features) tuple where file_path.shape == (numfeatues,),
        classes.shape == (numobs,) and features.shape == (numobs, numfeatures)

    """
    loaded = np.load(file_path)
    file_paths, classes, features = (
        loaded["file_paths"],
        loaded["classes"],
        loaded["features"],
    )
    assert (
        classes.shape[0] == features.shape[0]
    ), "Number of class observations does not match number of feature observations."
    assert (
        len(file_paths) == features.shape[1]
    ), "Number of files does not match number of features."
    return (file_paths, classes, features)
