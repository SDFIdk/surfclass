from osgeo import gdal
import numpy as np
from surfclass.noise import fill_nearest_neighbor


def test_fill_nearestneighbor(classraster_filepath):
    ds = gdal.Open(str(classraster_filepath))
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    masked_data = np.ma.array(data, mask=data == nodata)
    assert np.any(data == nodata), "Test data doesnt have nodata"
    filled = fill_nearest_neighbor(masked_data)
    assert not np.any(filled == nodata)
    assert not np.any(filled == masked_data.fill_value)
