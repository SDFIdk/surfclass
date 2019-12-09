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
    assert np.max(grid) == 60.963
    assert np.min(grid) == 2.809


def test_gridsampler_bbox(las_filepath):
    pl = lidar.open_pdal_pipeline(las_filepath)
    pl.execute()
    points = pl.arrays[0]
    assert len(points) == 16133
    bbox = (727000 - 0.8, 6171000 - 0.8, 728000 + 0.8, 6172000 + 0.8)
    resolution = 0.4
    sampler = lidar.GridSampler(points, bbox, resolution)

    grid = sampler.make_grid("Z", nodata=-999, masked=True)
    assert grid.shape == (2504, 2504)
