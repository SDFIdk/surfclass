import pytest
from osgeo import ogr, osr
import numpy as np
from surfclass import Bbox
from surfclass.rasterreader import RasterReader, MaskedRasterReader


def test_rasterreader(classraster_filepath):
    reader = RasterReader(classraster_filepath)
    assert reader
    assert isinstance(reader.srs, osr.SpatialReference)
    assert reader.bbox == Bbox(727000.0, 6171000.0, 728000.0, 6172000.0)
    pix_win = reader.bbox_to_pixel_window(reader.bbox)
    assert pix_win == (0, 0, 500, 500)
    pix_win = reader.bbox_to_pixel_window(
        Bbox(727500.0, 6171600.0, 727600.0, 6171750.0)
    )
    assert pix_win == (250, 125, 50, 75)
    # Read entire, unmasked
    data = reader.read_raster()
    assert data.dtype == "uint8"
    assert int(np.sum(data)) == 554094
    assert not np.ma.is_masked(data)
    assert int(np.sum(data == reader.nodata)) == 66724
    # Read entire, masked
    masked_data = reader.read_raster(masked=True)
    assert masked_data.dtype == "uint8"
    assert np.ma.is_masked(masked_data)
    assert int(np.sum(masked_data.compressed())) == 554094
    assert int(np.sum(masked_data.mask)) == 66724
    np.testing.assert_array_equal(data[data != reader.nodata], masked_data.compressed())
    # Read bbox, unmasked
    data_bbox = reader.read_raster(bbox=Bbox(727500.0, 6171600.0, 727600.0, 6171750.0))
    assert data_bbox.shape == (75, 50)
    assert int(np.sum(data_bbox)) == 8191
    assert not np.ma.is_masked(data_bbox)
    assert int(np.sum(data_bbox == reader.nodata)) == 1055
    # Read bbox masked
    masked_data_bbox = reader.read_raster(
        bbox=Bbox(727500.0, 6171600.0, 727600.0, 6171750.0), masked=True
    )
    assert masked_data_bbox.shape == (75, 50)
    assert np.ma.is_masked(masked_data_bbox)
    assert int(np.sum(masked_data_bbox.compressed())) == 8191
    assert int(np.sum(masked_data_bbox.mask)) == 1055
    np.testing.assert_array_equal(
        data_bbox[data_bbox != reader.nodata], masked_data_bbox.compressed()
    )
    # Read window, unmasked
    data_window = reader.read_raster(window=(23, 51, 27, 29))
    assert data_window.shape == (29, 27)
    assert int(np.sum(data_window)) == 1591
    assert not np.ma.is_masked(data_window)
    assert int(np.sum(data_window == reader.nodata)) == 154
    # Read bbox masked
    masked_data_window = reader.read_raster(window=(23, 51, 27, 29), masked=True)
    assert masked_data_window.shape == (29, 27)
    assert np.ma.is_masked(masked_data_window)
    assert int(np.sum(masked_data_window.compressed())) == 1591
    assert int(np.sum(masked_data_window.mask)) == 154
    np.testing.assert_array_equal(
        data_window[data_window != reader.nodata], masked_data_window.compressed()
    )
    # Test invalid spatial filters
    with pytest.raises(ValueError):
        reader.read_raster(window=(0, 0, 1, -1))
    with pytest.raises(ValueError):
        reader.read_raster(window=(0, 0, reader.width + 1, 1))
    with pytest.raises(ValueError):
        # Both are valid but cannot be specified at the same time
        reader.read_raster(
            window=(23, 51, 27, 29), bbox=Bbox(727500.0, 6171600.0, 727600.0, 6171750.0)
        )


def test_maskedrasterreader(classraster_filepath):
    reader = MaskedRasterReader(classraster_filepath)
    assert reader
    assert isinstance(reader.srs, osr.SpatialReference)
    assert reader.bbox == Bbox(727000.0, 6171000.0, 728000.0, 6172000.0)
    pix_win = reader.bbox_to_pixel_window(reader.bbox)
    assert pix_win == (0, 0, 500, 500)
    pix_win = reader.bbox_to_pixel_window(
        Bbox(727500.0, 6171600.0, 727600.0, 6171750.0)
    )
    assert pix_win == (250, 125, 50, 75)
    # Try using a point
    pnt = ogr.Geometry(ogr.wkbPoint)
    pnt.AddPoint(727500.0, 6171600.0)
    pnt.AssignSpatialReference(reader.srs)
    data = reader.read_masked(pnt)
    np.testing.assert_array_equal(data, np.ma.empty(shape=(0, 0)))
    # Now use a proper polygon
    poly = pnt.Buffer(100)
    data = reader.read_masked(poly)
    assert data.shape == (100, 100)
    # Check that we have a mask
    assert int(np.sum(data.mask)) == 2140
    # Check data without mask
    assert int(np.sum(data.data)) == 25412
    # Check data with mask. (Must be less than unmasked)
    assert int(np.sum(data.compressed())) == 20432
