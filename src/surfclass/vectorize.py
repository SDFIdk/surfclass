from osgeo import gdal, ogr, osr
import numpy as np
from surfclass import Bbox


# https://gist.github.com/AsgerPetersen/9642444

class_map = {1: "xxx", 2: "yyy", 3: "zzz"}


def bbox_to_ogr_polygon(bbox):
    xmin, ymin, xmax, ymax = bbox
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(xmin, ymin)
    ring.AddPoint(xmax, ymin)
    ring.AddPoint(xmax, ymax)
    ring.AddPoint(xmin, ymax)
    ring.AddPoint(xmin, ymin)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly


class MaskedRasterReader:
    """Reads part of a raster defined by a polygon into a masked 2D array"""

    def __init__(self, raster_path):
        self._ds = gdal.Open(str(raster_path), gdal.GA_ReadOnly)
        assert self._ds, "Could not open raster"
        self._band = self._ds.GetRasterBand(1)
        self._bbox = None
        self._srs = None
        self.geotransform = self._ds.GetGeoTransform()
        self.nodata = self._band.GetNoDataValue()

        # Memory drivers
        self._ogr_mem_drv = ogr.GetDriverByName("Memory")
        self._gdal_mem_drv = gdal.GetDriverByName("MEM")

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
        xmin, ymin, xmax, ymax = bbox
        originX, pixel_width, _, originY, _, pixel_height = self.geotransform
        x1 = int((xmin - originX) / pixel_width)
        x2 = int((xmax - originX) / pixel_width + 0.5)
        y1 = int((ymax - originY) / pixel_height)
        y2 = int((ymin - originY) / pixel_height + 0.5)
        xsize = x2 - x1
        ysize = y2 - y1
        return (x1, y1, xsize, ysize)

    def read_masked(self, geom):
        if not isinstance(geom, ogr.Geometry):
            raise TypeError("Must be OGR geometry")
        mem_type = geom.GetGeometryType()
        ogr_env = geom.GetEnvelope()
        geom_bbox = Bbox(ogr_env[0], ogr_env[2], ogr_env[1], ogr_env[3])
        src_offset = self.bbox_to_pixel_window(geom_bbox)
        if src_offset[2] <= 0 or src_offset[3] <= 0:
            return np.ma.empty(shape=(0, 0))
        src_array = self._band.ReadAsArray(*src_offset)
        # calculate new geotransform of the feature subset
        new_gt = (
            (self.geotransform[0] + (src_offset[0] * self.geotransform[1])),
            self.geotransform[1],
            0.0,
            (self.geotransform[3] + (src_offset[1] * self.geotransform[5])),
            0.0,
            self.geotransform[5],
        )

        # Create a temporary vector layer in memory
        mem_ds = self._ogr_mem_drv.CreateDataSource("out")
        mem_layer = mem_ds.CreateLayer("mem_lyr", geom.GetSpatialReference(), mem_type)
        mem_feature = ogr.Feature(mem_layer.GetLayerDefn())
        mem_feature.SetGeometry(geom)
        mem_layer.CreateFeature(mem_feature)

        # Rasterize the feature
        mem_raster_ds = self._gdal_mem_drv.Create(
            "", src_offset[2], src_offset[3], 1, gdal.GDT_Byte
        )
        mem_raster_ds.SetProjection(self._ds.GetProjection())
        mem_raster_ds.SetGeoTransform(new_gt)
        # Burn 1 inside our feature
        gdal.RasterizeLayer(mem_raster_ds, [1], mem_layer, burn_values=[1])
        rasterized_array = mem_raster_ds.ReadAsArray()

        # Mask the source data array with our current feature mask
        masked = np.ma.MaskedArray(src_array, mask=np.logical_not(rasterized_array))
        return masked


class StatsCalculator:
    def __init__(self, featurereader, maskedrasterreader, outputlayer, classes):
        self._featurereader = featurereader
        self._rasterreader = maskedrasterreader
        self._outlyr = outputlayer
        self._classmap = (
            classes if isinstance(classes, dict) else {x: f"class_{x}" for x in classes}
        )
        self._class_field_index = None
        self._total_field = "total_count"
        self._total_field_index = None
        self._zero_counts = {x: 0 for x in self._classmap}

    def process(self):
        self._add_fields()
        vdefn = self._outlyr.GetLayerDefn()
        for f in self._featurereader:
            all_classes = dict(self._zero_counts)
            class_counts = self._count_classes_inside(f.geometry())
            # Add zero counts for classes not seen
            all_classes.update(class_counts)
            outfeat = ogr.Feature(vdefn)
            outfeat.SetFrom(f)
            total_count = 0
            for class_id, count in all_classes.items():
                total_count += count
                try:
                    field_id = self._class_field_index[class_id]
                    outfeat.SetFieldInteger64(field_id, count)
                except KeyError:
                    pass
            outfeat.SetFieldInteger64(self._total_field_index, total_count)
            self._outlyr.CreateFeature(outfeat)

    def _add_fields(self):
        self._class_field_index = {}
        vdefn = self._outlyr.GetLayerDefn()
        for class_id, name in self._classmap.items():
            # has the field been created already?
            field_index = vdefn.GetFieldIndex(name)
            if field_index < 0:
                # Create field
                fd = ogr.FieldDefn(name, ogr.OFTInteger64)
                self._outlyr.CreateField(fd)
                field_index = vdefn.GetFieldIndex(name)
            assert field_index >= 0, "Could not create field %s" % name
            self._class_field_index[class_id] = field_index
        # Field for total count
        field_index = vdefn.GetFieldIndex(self._total_field)
        if field_index < 0:
            # Create field
            fd = ogr.FieldDefn(self._total_field, ogr.OFTInteger64)
            self._outlyr.CreateField(fd)
            self._total_field_index = vdefn.GetFieldIndex(self._total_field)
            field_index = vdefn.GetFieldIndex(self._total_field)
        assert field_index >= 0, "Could not create field %s" % self._total_field

    def _count_classes_inside(self, geom):
        masked_data = self._rasterreader.read_masked(geom)
        # Ok, now count classes (including nodata):
        unique, counts = np.unique(masked_data.compressed(), return_counts=True)
        class_counts = dict(zip(unique, counts))
        return class_counts


class FeatureReader:
    def __init__(self, datasource, layer=None):
        self.ds = (
            datasource
            if isinstance(datasource, ogr.DataSource)
            else ogr.Open(str(datasource), gdal.GA_ReadOnly)
        )
        assert self.ds, "Could not open OGR datasource: %s" % datasource
        self.lyr = (
            layer
            if isinstance(layer, ogr.Layer)
            else (self.ds.GetLayerByName(layer) if layer else self.ds.GetLayer(0))
        )
        assert self.lyr, "Could not open datasource layer %s" % layer
        self.schema = self.lyr.GetLayerDefn()
        self.srs = self.schema.GetGeomFieldDefn(0).srs
        self._clip = False
        self._bbox_filter = None
        self._clip_geom = None

    def set_bbox_filter(self, bbox, clip=False):
        self._clip = clip
        self._bbox_filter = Bbox(*bbox) if bbox else None
        if not bbox:
            # Unset filter
            self.lyr.SetSpatialFilter(None)
        else:
            self.lyr.SetSpatialFilterRect(*bbox)
            if clip:
                self._clip_geom = bbox_to_ogr_polygon(bbox)
                self._clip_geom.AssignSpatialReference(self.srs)
        self.reset_reading()

    def reset_reading(self):
        self.lyr.ResetReading()

    def __iter__(self):
        return self

    def __next__(self):
        feat = self.lyr.GetNextFeature()
        if feat is None:
            raise StopIteration
        if self._clip:
            intersection = feat.geometry().Intersection(self._clip_geom)
            if intersection.IsEmpty():
                return self.__next__()
            feat.SetGeometryDirectly(intersection)
        return feat


def open_or_create_destination_datasource(dst_ds_name, dst_format=None, dsco=None):
    dst_ds = ogr.Open(dst_ds_name, update=1)
    if dst_ds is None:
        dst_ds = ogr.Open(dst_ds_name)
        if dst_ds is not None:
            raise Exception(
                'Output datasource "%s" exists, but cannot be opened in update mode'
                % dst_ds
            )
        drv = ogr.GetDriverByName(dst_format)
        if drv is None:
            raise Exception("Cannot find driver %s" % dst_format)
        dst_ds = drv.CreateDataSource(dst_ds_name, options=dsco or [])
        if dst_ds is None:
            raise Exception("Cannot create datasource '%s'" % dst_ds_name)
    return dst_ds


def open_or_create_similar_layer(src_lyr, dst_ds, dst_lyr_name=None, lco=None):
    if dst_lyr_name is None:
        count = dst_ds.GetLayerCount()
        if count > 1:
            raise Exception(
                "Destination datasource has multiple layers. Layername must be specified"
            )
        if count == 1:
            lyr = dst_ds.GetLayer(0)
        else:
            lyr = dst_ds.CreateLayer(
                "surfclass", src_lyr.GetSpatialRef(), src_lyr.GetGeomType(), lco or []
            )
            copy_fields(src_lyr, lyr)
    else:
        lyr = dst_ds.GetLayer(dst_lyr_name)
        if lyr is None:
            lyr = dst_ds.CreateLayer(
                dst_lyr_name, src_lyr.GetSpatialRef(), src_lyr.GetGeomType(), lco or []
            )
            if lyr is None:
                raise Exception('Could not create layer "%s"' % dst_lyr_name)
            copy_fields(src_lyr, lyr)
    return lyr


def copy_fields(src_lyr, dst_lyr):
    layer_defn = src_lyr.GetLayerDefn()
    for idx in range(layer_defn.GetFieldCount()):
        fld_defn = layer_defn.GetFieldDefn(idx)
        if dst_lyr.CreateField(fld_defn) != 0:
            raise Exception(
                'Cannot create field "%s" in layer "%s"'
                % (fld_defn.GetName(), dst_lyr.GetName())
            )
