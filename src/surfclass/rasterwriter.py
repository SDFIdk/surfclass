from osgeo import gdal, osr
import numpy as np

gdal_int_options = ["TILED=YES", "COMPRESS=deflate", "PREDICTOR=2"]
gdal_float_options = ["TILED=YES", "COMPRESS=deflate", "PREDICTOR=3"]


def write_to_file(filename, array, origin, resolution, epsg_code, nodata=None):
    cols, rows = array.shape[1], array.shape[0]
    originX, originY = origin
    dtype = array.dtype
    gdal_type = dtype_to_gdaltype(dtype)
    gdal_options = gdaltype_to_creationoptions(gdal_type)
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(filename, cols, rows, 1, gdal_type, options=gdal_options)
    ds.SetGeoTransform((originX, resolution, 0, originY, 0, -1 * resolution))
    band = ds.GetRasterBand(1)

    # TODO Handle the case where nodata is none (best guess to avoid removing information)
    if nodata is not None:
        band.SetNoDataValue(nodata)
        if isinstance(array, np.ma.MaskedArray):
            array = array.filled(fill_value=nodata)

    band.WriteArray(array)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_code)
    ds.SetProjection(srs.ExportToWkt())
    band.FlushCache()
    ds = None


map_dtype_gdal = {
    "uint8": gdal.GDT_Byte,
    "uint16": gdal.GDT_UInt16,
    "int16": gdal.GDT_Int16,
    "uint32": gdal.GDT_UInt32,
    "int32": gdal.GDT_Int32,
    "uint64": gdal.GDT_Float64,
    "int64": gdal.GDT_Float64,
    "float32": gdal.GDT_Float32,
    "float64": gdal.GDT_Float64,
}


def gdaltype_to_creationoptions(gdaltype):
    if gdaltype in [
        gdal.GDT_Byte,
        gdal.GDT_Int16,
        gdal.GDT_Int32,
        gdal.GDT_UInt16,
        gdal.GDT_UInt32,
    ]:
        return gdal_int_options
    if gdaltype in [gdal.GDT_Float32, gdal.GDT_Float64]:
        return gdal_float_options
    raise NotImplementedError()


def dtype_to_gdaltype(dtype):
    t = str(dtype)
    return map_dtype_gdal[t]
