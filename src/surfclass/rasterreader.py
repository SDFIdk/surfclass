# pylint: disable=R0916
from osgeo import gdal, ogr, osr
import numpy as np
from surfclass import Bbox


class RasterReader:
    def __init__(self, raster_path):
        self._ds = gdal.Open(str(raster_path), gdal.GA_ReadOnly)
        assert self._ds, "Could not open raster"
        self._band = self._ds.GetRasterBand(1)
        self._bbox = None
        self._srs = None
        self.geotransform = self._ds.GetGeoTransform()
        self.resolution = self.geotransform[1]
        self.nodata = self._band.GetNoDataValue()
        self.width = self._ds.RasterXSize
        self.height = self._ds.RasterYSize
        self.shape = (self.height, self.width)

        # Memory drivers
        self._ogr_mem_drv = ogr.GetDriverByName("Memory")
        self._gdal_mem_drv = gdal.GetDriverByName("MEM")

        # We do not support rotated rasters
        assert (
            self.geotransform[2] == self.geotransform[4] == 0
        ), "Rotated rasters are not supported"

    @property
    def srs(self):
        if self._srs is None:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(self._ds.GetProjection())
            self._srs = srs
        return self._srs

    @property
    def bbox(self):
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
        """Returns pixel window for a Bbox in world coordinates

        No checks are made regarding bbox being a subset of raster extent. Bbox coordinates MUST
        be in the same spatial reference system as used for the raster georeferencing.

        Args:
            bbox (Bbox): Part of raster in raster world coordinates (xmin, ymin, xmax, ymax).

        Returns:
            tuple: (column, row, numcolums, numrows) in pixels
        """
        xmin, ymin, xmax, ymax = bbox
        originX, pixel_width, _, originY, _, pixel_height = self.geotransform
        x1 = int((xmin - originX) / pixel_width)
        x2 = int((xmax - originX) / pixel_width + 0.5)
        y1 = int((ymax - originY) / pixel_height)
        y2 = int((ymin - originY) / pixel_height + 0.5)
        xsize = x2 - x1
        ysize = y2 - y1
        return (x1, y1, xsize, ysize)

    def window_geotransform(self, window):
        """Calculates geotransform for raster subset expressed as a window

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
        """Read (part of) raster and return as masked or raw numpy array

        Reads entire raster if neither bbox nor window is given.

        Bbox nad window must be subset of raster extent. Bbox coordinates MUST
        be in the same spatial reference system as used for the raster georeferencing.

        Args:
            window (tuple, optional): Part of raster to read expressed as a pixel window
                (column, row, numcolums, numrows). Defaults to None.
            bbox (Bbox, optional): Part of raster to read expressed in world coordinates
                (xmin, ymin, xmax, ymax). Defaults to None.
            masked (bool, optional): Return a MaskedArray masked by the raster nodatavalue. Defaults to False.

        Returns:
            ndarray: 2D ndarray (possibly masked)
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
            or self.width < col
            or row < 0
            or self.height < row
            or cols < 0
            or self.width < col + cols
            or rows < 0
            or self.height < row + rows
        ):
            raise ValueError(f"Window outside raster requested. Window: {src_offset}")

        src_array = self._band.ReadAsArray(*src_offset)

        if src_offset[2] <= 0 or src_offset[3] <= 0:
            if masked:
                return np.ma.empty(shape=(0, 0))
            return np.empty(shape=(0, 0))

        if masked:
            return np.ma.masked_values(src_array, self.nodata)

        return src_array


class MaskedRasterReader(RasterReader):
    """Reads part of a raster defined by a polygon into a masked 2D array"""

    # TODO: rename to something with geom and refactor
    def read_masked(self, geom):
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
