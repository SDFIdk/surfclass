from osgeo import gdal
import numpy as np
import pytest
from surfclass.noise import fill_nearest_neighbor, sieve_mask, sieve, majority_vote


def read_masked(f):
    ds = gdal.Open(str(f))
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    return np.ma.masked_values(data, nodata)


def test_fill_nearestneighbor(classraster_filepath):
    masked_data = read_masked(classraster_filepath)
    filled = fill_nearest_neighbor(masked_data)
    assert not isinstance(filled, np.ma.MaskedArray)
    assert not np.any(filled == masked_data.fill_value)


def test_sieve_mask(classraster_filepath):
    masked_data = read_masked(classraster_filepath)
    mask = sieve_mask(masked_data, 1, 5)
    assert int(np.sum(mask)) == 4032


def test_sieve(classraster_filepath):
    masked_data = read_masked(classraster_filepath)
    assert int(np.sum(masked_data.mask)) == 66724, "Test file changed"
    sieve(masked_data, 5)
    assert np.sum(masked_data.mask) > 66724, "Sieve did not modify mask"
    for c in range(6):
        mask = sieve_mask(masked_data, c, 5)
        assert int(np.sum(mask)) == 0, "Sieve didnt remove all small clusters"


def test_majority_vote(classraster_filepath):
    masked_data = read_masked(classraster_filepath)
    filtered = majority_vote(masked_data)
    assert isinstance(filtered, np.ma.MaskedArray)
    assert masked_data.dtype == filtered.dtype
    values, counts = np.unique(filtered, return_counts=True)
    np.testing.assert_array_equal(values.compressed(), [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(counts, [57524, 42, 18721, 112611, 250, 60852])
    # Test 4 connectedness
    filtered = majority_vote(
        masked_data, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    )
    values, counts = np.unique(filtered, return_counts=True)
    np.testing.assert_array_equal(values.compressed(), [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(counts, [57128, 48, 23977, 109523, 355, 58969])
    # Test iterations
    filtered = majority_vote(masked_data, iterations=5)
    values, counts = np.unique(filtered, return_counts=True)
    np.testing.assert_array_equal(values.compressed(), [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(counts, [62494, 35, 12850, 115733, 97, 58791])
    # Test throw
    int64_data = masked_data.astype("int64")
    with pytest.raises(Exception):
        filtered = majority_vote(int64_data, iterations=5)
