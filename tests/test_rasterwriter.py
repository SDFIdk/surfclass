import os
import numpy as np
from osgeo import gdal
from surfclass import rasterwriter


def test_writer(tmp_path):
    data = np.arange(1500).astype("float32").reshape((30, 50))
    origin = (550000, 6150000)
    resolution = 1
    epsg = 25832
    outfile = os.path.join(tmp_path, "test_writer.tif")
    rasterwriter.write_to_file(outfile, data, origin, resolution, epsg)
    assert os.path.exists(outfile)
    ds = gdal.Open(outfile)
    assert ds.GetGeoTransform() == (
        origin[0],
        resolution,
        0,
        origin[1],
        0,
        -1 * resolution,
    )
    assert ds.RasterXSize == data.shape[1]
    assert ds.RasterYSize == data.shape[0]
    band = ds.GetRasterBand(1)
    assert band.DataType == gdal.GDT_Float32
    ds = None
