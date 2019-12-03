import os
import numpy as np
from osgeo import gdal
from surfclass import rasterwriter
from surfclass.rasterreader import RasterReader


def test_writer(tmp_path):
    data = np.arange(1500).astype("float32").reshape((30, 50))
    origin = (550000, 6150000)
    resolution = 1
    epsg = 25832
    outfile = os.path.join(tmp_path, "test_writer.tif")
    rasterwriter.write_to_file(outfile, data, origin, resolution, epsg)
    assert os.path.exists(outfile)
    reader = RasterReader(outfile)
    assert reader.geotransform == (
        origin[0],
        resolution,
        0,
        origin[1],
        0,
        -1 * resolution,
    )
    assert reader.width == data.shape[1]
    assert reader.height == data.shape[0]
    assert reader.shape == data.shape
    read_data = reader.read_raster()
    assert read_data.dtype == "float32"
    assert reader.nodata is None


def test_writernodata(tmp_path):
    data = np.arange(1500).astype("float32").reshape((30, 50))
    data[10:15, 10:15] = 0
    origin = (550000, 6150000)
    resolution = 1
    epsg = 25832
    outfile = os.path.join(tmp_path, "test_writer_nodata.tif")
    rasterwriter.write_to_file(outfile, data, origin, resolution, epsg, nodata=0)
    assert os.path.exists(outfile)
    reader = RasterReader(outfile)
    assert reader.geotransform == (
        origin[0],
        resolution,
        0,
        origin[1],
        0,
        -1 * resolution,
    )
    assert reader.width == data.shape[1]
    assert reader.height == data.shape[0]
    assert reader.shape == data.shape
    read_data = reader.read_raster()
    assert read_data.dtype == "float32"
    assert reader.nodata == 0
    assert int(np.sum(reader.read_raster(masked=True).mask)) == 26

    masked = np.ma.array(data)
    masked.mask = data == 0
    outfile = os.path.join(tmp_path, "test_writer_nodata.tif")
    rasterwriter.write_to_file(outfile, masked, origin, resolution, epsg)
    assert os.path.exists(outfile)
    reader = RasterReader(outfile)
    assert reader.geotransform == (
        origin[0],
        resolution,
        0,
        origin[1],
        0,
        -1 * resolution,
    )
    assert reader.width == data.shape[1]
    assert reader.height == data.shape[0]
    assert reader.shape == data.shape
    read_data = reader.read_raster()
    assert read_data.dtype == "float32"
    assert reader.nodata is not None
    assert int(np.sum(reader.read_raster(masked=True).mask)) == 26
