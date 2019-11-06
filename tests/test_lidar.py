import numpy as np
from surfclass import lidar


def test_open_pipeline(las_filepath):
    pl = lidar.open_pdal_pipeline(las_filepath)
    assert pl
    pl.execute()
    assert len(pl.arrays) == 1
    assert len(pl.arrays[0]) == 16133


def test_gridsampler(las_filepath):
    pl = lidar.open_pdal_pipeline(las_filepath)
    pl.execute()
    points = pl.arrays[0]
    assert len(points) == 16133
    bbox = (727000, 6171000, 728000, 6172000)
    resolution = 10
    sampler = lidar.GridSampler(points, bbox, resolution)

    grid = sampler.make_grid("Z", nodata=-999, masked=True)
    assert grid.shape == (100, 100)
    assert np.max(grid) == 60.99
    assert np.min(grid) == 38.097
