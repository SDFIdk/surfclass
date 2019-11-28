from osgeo import gdal
import numpy as np
from surfclass.noise import fill_nearest_neighbor, sieve_mask, sieve


def test_fill_nearestneighbor(classraster_filepath):
    ds = gdal.Open(str(classraster_filepath))
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    masked_data = np.ma.array(data, mask=data == nodata)
    assert np.any(data == nodata), "Test data doesnt have nodata"
    filled = fill_nearest_neighbor(masked_data)
    assert not isinstance(filled, np.ma.MaskedArray)
    assert not np.any(filled == nodata)
    assert not np.any(filled == masked_data.fill_value)


def test_sieve_mask(classraster_filepath):
    ds = gdal.Open(str(classraster_filepath))
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    masked_data = np.ma.array(data, mask=data == nodata)
    assert np.any(data == nodata), "Test data doesnt have nodata"
    mask = sieve_mask(masked_data, 1, 5)
    assert int(np.sum(mask)) == 4032


def test_sieve(classraster_filepath):
    ds = gdal.Open(str(classraster_filepath))
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    masked_data = np.ma.array(data, mask=data == nodata)
    assert np.any(data == nodata), "Test data doesnt have nodata"
    assert int(np.sum(masked_data.mask)) == 66724, "Test file changed"
    sieve(masked_data, 5)
    assert np.sum(masked_data.mask) > 66724, "Sieve did not modify mask"
    for c in range(6):
        mask = sieve_mask(masked_data, c, 5)
        assert int(np.sum(mask)) == 0, "Sieve didnt remove all small clusters"
