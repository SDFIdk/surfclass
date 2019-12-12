"""IO for raster files."""
# pylint: disable=R0916
import logging
from osgeo import gdal, ogr, osr
import numpy as np
from surfclass import Bbox

logger = logging.getLogger(__name__)

gdal_int_options = ["TILED=YES", "COMPRESS=deflate"]
gdal_float_options = ["TILED=YES", "COMPRESS=deflate"]


class RasterReader:
    """Reads one band raster file into numpy arrays."""

    def __init__(self, raster_path):
        """Create instance of RasterReader.

        Args:
            raster_path (str): Path to raster file

        """
        self.raster_path = raster_path
        self._ds = gdal.Open(str(raster_path), gdal.GA_ReadOnly)
        assert self._ds, "Could not open raster"
        self._band = self._ds.GetRasterBand(1)
        self._bbox = None
        self._srs = None
        #: tuple: Raster geotransform.
        self.geotransform = self._ds.GetGeoTransform()
        #: float: Cell size in srs units.
        self.resolution = self.geotransform[1]
        #: float, None: Raster value indicating nodata cells.
        self.nodata = self._band.GetNoDataValue()
        #: int: Raster width in cells.
        self.width = self._ds.RasterXSize
        #: int: Raster height in cells.
        self.height = self._ds.RasterYSize
        #: tuple: Raster shape (rows, columns).
        self.shape = (self.height, self.width)

        # Memory drivers
        self._ogr_mem_drv = ogr.GetDriverByName("Memory")
        self._gdal_mem_drv = gdal.GetDriverByName("MEM")

        # We do not support rotated rasters
        assert (
            self.geotransform[2] == self.geotransform[4] == 0
        ), "Rotated rasters are not supported"

        logger.debug(
            "Opened: '%s'. Geotransform: %s. Nodata: %s. Shape: %s",
            raster_path,
            self.geotransform,
            self.nodata,
            self.shape,
        )

    @property
    def srs(self):
        """SpatialReference: Spatial reference system used by the loaded raster file."""
        if self._srs is None:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(self._ds.GetProjection())
            self._srs = srs
        return self._srs

    @property
    def bbox(self):
        """Bbox: Bounding box of loaded raster."""
        if self._bbox is None:
            xmin = self.geotransform[0]
            ymax = self.geotransform[3]
            width = self.geotransform[1] * self._ds.RasterXSize
            height = self.geotransform[5] * self._ds.RasterYSize
            xmax = xmin + width
            ymin = ymax + height
            self._bbox = Bbox(xmin, ymin, xmax, ymax)
        return self._bbox

    def bbox_to_pixel_window(self, bbox):
        """Returns pixel window for a Bbox in world coordinates.

        No checks are made regarding bbox being a subset of raster extent. Bbox coordinates MUST
        be in the same spatial reference system as used for the raster georeferencing.

        Args:
            bbox (Bbox): Part of raster in raster world coordinates (xmin, ymin, xmax, ymax).

        Returns:
            tuple: (column, row, numcolums, numrows) in pixels

        """
        xmin, ymin, xmax, ymax = bbox
        originX, pixel_width, _, originY, _, pixel_height = self.geotransform
        x1 = (xmin - originX) / pixel_width
        x2 = (xmax - originX) / pixel_width
        y1 = (ymax - originY) / pixel_height
        y2 = (ymin - originY) / pixel_height
        # Convert float pix indices to int pix indices same way as GDAL:
        # https://github.com/OSGeo/gdal/blob/2df8e8105a51b47e911253a4d6bf48dab8ac5bd1/gdal/apps/gdal_translate_lib.cpp#L878-L892
        x1 = int(x1 + 0.001)
        y1 = int(y1 + 0.001)
        x2 = int(x2 + 0.5)
        y2 = int(y2 + 0.5)
        xsize = x2 - x1
        ysize = y2 - y1
        return (x1, y1, xsize, ysize)

    def window_geotransform(self, window):
        """Calculates geotransform for raster subset expressed as a window.

        Args:
            window (tuple): (column, row, numcolums, numrows) in pixels
        Returns:
            tuple: (originX, pixel_width, 0, originY, 0, pixel_height)

        """
        return (
            (self.geotransform[0] + (window[0] * self.geotransform[1])),
            self.geotransform[1],
            0.0,
            (self.geotransform[3] + (window[1] * self.geotransform[5])),
            0.0,
            self.geotransform[5],
        )

    def read_raster(self, window=None, bbox=None, masked=False):
        """Read (part of) raster and return as masked or raw numpy array.

        Reads entire raster if neither bbox nor window is given.

        Bbox coordinates MUST be in the same spatial reference system as used for the raster georeferencing.

        Args:
            window (tuple, optional): Part of raster to read expressed as a pixel window
                (column, row, numcolums, numrows). Defaults to None.
            bbox (Bbox, optional): Part of raster to read expressed in world coordinates
                (xmin, ymin, xmax, ymax). Defaults to None.
            masked (bool, optional): Return a MaskedArray masked by the raster nodatavalue. Defaults to False.

        Returns:
            ndarray: 2D ndarray (possibly masked)

        Raises:
            ValueError: If requested `bbox` or `window` is outside raster coverage.
            ValueError: If both `bbox` and `window` are specified.

        """
        if bbox and window:
            raise ValueError("Only one of window and bbox can be specified")
        if window:
            src_offset = window
        elif bbox:
            src_offset = self.bbox_to_pixel_window(bbox)
        else:
            src_offset = (0, 0, self._ds.RasterXSize, self._ds.RasterYSize)

        col, row, cols, rows = src_offset
        if (
            col < 0
            or self.width <= col
            or row < 0
            or self.height <= row
            or cols < 0
            or self.width < col + cols
            or rows < 0
            or self.height < row + rows
        ):
            raise ValueError(f"Window outside raster requested. Window: {src_offset}")

        logger.debug("Reading window: %s", src_offset)
        src_array = self._band.ReadAsArray(*src_offset)

        if src_offset[2] <= 0 or src_offset[3] <= 0:
            if masked:
                return np.ma.empty(shape=(0, 0))
            return np.empty(shape=(0, 0))

        if masked:
            return (
                np.ma.array(src_array)
                if self.nodata is None
                else np.ma.masked_values(src_array, self.nodata)
            )

        return src_array


class MaskedRasterReader(RasterReader):
    """Reads part of a raster defined by a polygon into a 2D MaskedArray with a mask marking cells outside the polygon."""

    def read_2d(self, geom):
        """Reads part of the raster into a 2D MaskedArray with a mask marking cells outside the polygon.

        This is suitable for doing analysis on cells inside a given geometry object.

        Note: The mask marks cells outside the geometry. This means that cells inside the geometry are NOT masked
            even if they are equal to the raster nodata value.

        Args:
            geom (osgeo.ogr.Geometry): OGR Geometry object

        Raises:
            TypeError: If geometry is not an `osgeo.ogr.Geometry`
            ValueError: If bbox of the geometry is entirely or partly outside raster coverage.

        Returns:
            [numpy.ma.maskedArray]: A masked array where cells outside the geometry are masked.

        """
        if not isinstance(geom, ogr.Geometry):
            raise TypeError("Must be OGR geometry")
        mem_type = geom.GetGeometryType()
        ogr_env = geom.GetEnvelope()
        geom_bbox = Bbox(ogr_env[0], ogr_env[2], ogr_env[1], ogr_env[3])
        window = self.bbox_to_pixel_window(geom_bbox)
        if window[2] <= 0 or window[3] <= 0:
            return np.ma.empty(shape=(0, 0))
        src_array = self.read_raster(window=window, masked=False)
        # calculate new geotransform of the feature subset
        new_gt = self.window_geotransform(window)

        # Create a temporary vector layer in memory
        mem_ds = self._ogr_mem_drv.CreateDataSource("out")
        mem_layer = mem_ds.CreateLayer("mem_lyr", geom.GetSpatialReference(), mem_type)
        mem_feature = ogr.Feature(mem_layer.GetLayerDefn())
        mem_feature.SetGeometry(geom)
        mem_layer.CreateFeature(mem_feature)

        # Rasterize the feature
        mem_raster_ds = self._gdal_mem_drv.Create(
            "", window[2], window[3], 1, gdal.GDT_Byte
        )
        mem_raster_ds.SetProjection(self._ds.GetProjection())
        mem_raster_ds.SetGeoTransform(new_gt)
        # Burn 1 inside our feature
        gdal.RasterizeLayer(mem_raster_ds, [1], mem_layer, burn_values=[1])
        rasterized_array = mem_raster_ds.ReadAsArray()

        # Mask the source data array with our current feature mask
        masked = np.ma.MaskedArray(src_array, mask=np.logical_not(rasterized_array))
        return masked

    def read_flattened(self, geom):
        """Read data within the geom into a 1D masked array.

        Output 1D array contains all cell values within the geom. It is a MaskedArray where the mask
        indicates cells with `nodata` value.

        Args:
            geom (osgeo.ogr.Geometry): OGR Geometry object.

        Raises:
            TypeError: If geometry is not an `osgeo.ogr.Geometry`
            ValueError: If bbox of the geometry is entirely or partly outside raster coverage.

        Returns:
            [numpy.ma.maskedArray]: A 1D masked array where cells with `nodata` value are masked.

        """
        # mask marks cells outside geom
        masked_subset = self.read_2d(geom)
        # 1D array with cells inside geom only
        flattened = masked_subset.compressed()
        # Mark nodata cells as masked
        return (
            np.ma.array(flattened)
            if self.nodata is None
            else np.ma.masked_values(flattened, self.nodata)
        )


def write_to_file(filename, array, origin, resolution, srs, nodata=None):
    """Writes a georeferenced ndarray to a geotiff file.

    This method uses a simple heurestic to choose output datatype. Best results are obtained when the dtype
    of the input array is as narrow as possible. For instance use 'uint8' for values in range(135).

    Args:
        filename (str): Path to write geotiff
        array (ndarray): 2D ndarray optionally a MaskedArray
        origin (tuple): World coordinates of upper left corner of upper left pixel (origin_x, origin_y)
        resolution (float): Pixel size in world coordinate units. Pixel width and height must be equal.
        srs (int or SpatialReference): Reference system of supplied origin coordinates. Either an EPSG
            code specified as an int or an entire SpatialReference object.
        nodata (number, optional): Pixel value to set as nodatavalue in output raster. Defaults to None.

    """
    cols, rows = array.shape[1], array.shape[0]
    originX, originY = origin
    dtype = array.dtype
    gdal_type = dtype_to_gdaltype(dtype)
    gdal_options = gdaltype_to_creationoptions(gdal_type)
    geotransform = (originX, resolution, 0, originY, 0, -1 * resolution)

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(filename, cols, rows, 1, gdal_type, options=gdal_options)
    ds.SetGeoTransform(geotransform)
    band = ds.GetRasterBand(1)

    if nodata is None and np.ma.is_masked(array):
        nodata = find_nodata_value(array)

    if nodata is not None:
        band.SetNoDataValue(nodata)
        if np.ma.is_masked(array):
            array = array.filled(fill_value=nodata)

    if isinstance(srs, int):
        epsg_code = srs
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg_code)
    if not isinstance(srs, osr.SpatialReference):
        raise ValueError("srs must be either EPSG code or a SpatialReference object")

    logger.debug(
        "Writing file '%s'. Geotransform: %s. Nodata: %s",
        filename,
        geotransform,
        nodata,
    )
    band.WriteArray(array)
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
    """Gets GDAL creation options suitable for a GDAL datatype.

    Args:
        gdaltype (int): GDAL datatype value. For instance `osgeo.gdal.GDT_Byte`

    Raises:
        NotImplementedError: If given GDAL data type is not supported.

    Returns:
        list of str: List of GDAL creation options.

    """
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
    """Gets a GDAL datatype which matches a numpy.dtype.

    Args:
        dtype (numpy.dtype): Numpy datatype

    Returns:
        int: GDAL datatype value.

    """
    t = str(dtype)
    return map_dtype_gdal[t]


def find_nodata_value(a):
    """Tries to find a usable nodata value.

    Args:
        a (ndarray): 2D ndarray

    Returns:
        [number]: A number which is not present in the array and which is representable in the array datatype

    """
    amin, amax = np.ma.min(a), np.ma.max(a)
    t = a.dtype
    if t.kind == "f":
        tinfo = np.finfo(t)
    if t.kind in ("i", "u"):
        tinfo = np.iinfo(t)
    nines = [-99, -999, -9999, -99999, -999999, -9999999, -99999999, -999999999]
    n = next(x for x in nines if tinfo.min < x < amin)
    if n:
        return n
    if amin > 0:
        return 0
    n = next(-x for x in nines if amax < -x < tinfo.max)
    if n:
        return n
    if tinfo.min < amin:
        return tinfo.min
    if tinfo.max > amax:
        return tinfo.max
    raise Exception("No suitable nodata value found")
