"""Tools for classifying data."""
import numpy as np
from surfclass import rasterio


def stack_rasters(raster_paths, bbox=None):
    """Convert list of raster paths to arrays and stack them along the 3rd axis.

    If a bbox is supplied the whole stack will be read using that bbox, otherwise the
    whole raster is read. All rasters must have equal geotransformations to be considered valid.

    Args:
        raster_paths (list of str): List of paths to feature rasters. The order is important.
        bbox (tuple): Bounding Box of form (xmin,ymin,xmax,ymax)

    Returns:
        np.ma.ndarray: Masked 3D ndarray in the form (x,y,n) where n is the raster band
        np.ndarray: inverted mask, used for retrieving indices of valid cells
        tuple: Gdal Geotransform (x_min, pixel_size, 0, y_max, 0, -pixel_size)
        osgeo.osr.SpatialReference: srs Spatial reference system for the output raster (retrived from common srs in raster_paths).
        tuple: _shape, 2D shape of the resulting output array

    """
    features = []
    _tmp_geotransform = None
    srs = None
    _shape = None
    for f in raster_paths:
        rr = rasterio.RasterReader(f)

        if bbox is None:
            bbox = rr.bbox

        nodata = rr.nodata
        window = rr.bbox_to_pixel_window(bbox)
        geotransform = rr.window_geotransform(window)
        srs = rr.srs
        if _tmp_geotransform is None:
            _tmp_geotransform = geotransform

        assert np.allclose(
            _tmp_geotransform, geotransform
        ), "Features does not stack, geotransformations must be equal for all rasters"

        # Do not mask the raster.
        array = rr.read_raster(window=window, masked=False)
        _shape = array.shape
        # TODO: Continuing issue. Come up with common way to treat this
        if nodata is not None:
            array = np.ma.masked_values(array, nodata)
        else:
            array = np.ma.array(array)

        features.append(array)

    # Stack the features along the 3rd axis and reshape to (X,n)
    stacked_features = np.ma.dstack(features).reshape(-1, len(features))

    # Invert the mask to get all valid data points
    logical_or_mask = np.invert(stacked_features.mask.any(axis=1))

    # Get the data from the masked array
    valid_features = stacked_features[logical_or_mask].data

    # Return the intersected mask to be able insert nodata after classification
    return (valid_features, logical_or_mask, geotransform, srs, _shape)
