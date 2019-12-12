"""Tools for handling vector data."""
import logging
from osgeo import gdal, ogr
import numpy as np
from surfclass import Bbox

logger = logging.getLogger(__name__)


def bbox_to_ogr_polygon(bbox):
    """Convert a Bbox to a `Polygon` `osgeo.ogr.Geometry`.

    Args:
        bbox (tuple): (xmin, ymin, xmax, ymax)

    Returns:
        osgeo.ogr.Geometry: Polygon

    """
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


class ClassCounter:
    """Counts number of cells of a given value inside each `Polygon` from a vector feature datasource.

    Takes features from an input feature datasource one by one, reads the cells within this feature from
    the `maskedrastereader` counts the number of occurences of each cell value, adds these counts as attributes
    to the feature and writes the feature to `outputlayer`. Also adds the total number of cells within the
    feature as an attribute.

    Only values in `classes` are reported seperately.
    """

    def __init__(self, featurereader, maskedrasterreader, outputlayer, classes):
        """Inits a ClassCounter.

        Note:
            Features read from `featurereader` MUST be within the bbox of `maskedrasterreader`.
            Consider configuring the `featurereader` to clip the features.

        Args:
            featurereader (FeatureReader): Featurereader initialized with the feature datasource.
            maskedrasterreader (MaskedRasterReader): MaskedRasterReader initialized with the raster.
            outputlayer (osgeo.ogr.Layer): Layer to output features with added attributes.
            classes (list or dict): Class values to add as individual attributes. Either a list of class values
                or a dict(int, str) which maps class values to class names.

        """
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
        logger.debug("ClassCounter init")

    def process(self):
        """Start processing."""
        logger.debug("Started processing")
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
            logger.debug("Added field: '%s", name)
        # Field for total count
        field_index = vdefn.GetFieldIndex(self._total_field)
        if field_index < 0:
            # Create field
            fd = ogr.FieldDefn(self._total_field, ogr.OFTInteger64)
            self._outlyr.CreateField(fd)
            self._total_field_index = vdefn.GetFieldIndex(self._total_field)
            field_index = vdefn.GetFieldIndex(self._total_field)
            logger.debug("Added field: '%s", self._total_field)
        assert field_index >= 0, "Could not create field %s" % self._total_field

    def _count_classes_inside(self, geom):
        masked_data = self._rasterreader.read_2d(geom)
        # Ok, now count classes (including nodata):
        unique, counts = np.unique(masked_data.compressed(), return_counts=True)
        class_counts = dict(zip(unique, counts))
        return class_counts


class FeatureReader:
    """Read features from a vector feature datasource.

    Optionally filtering and clipping with a bbox.
    """

    def __init__(self, datasource, layer=None):
        """Init a FeatureReader.

        Args:
            datasource (str or osgeo.ogr.DataSource): An OGR datasource. Either defined by its datasource string or a
                `osgeo.ogr.DataSource` object.
            layer (str or osgeo.ogr.Layer, optional): Layer within the `datasource` to read from. Either a layer name,
                an `osgeo.ogr.Layer` object or `None`. If `None` the first layer in datasource is used. Defaults to None.

        Raises:
            AssertionError: If `datasource` cannot be opened.
            AssertionError: If `layer` cannot be opened.

        """
        #: osgeo.ogr.DataSource: DataSource object.
        self.ds = (
            datasource
            if isinstance(datasource, ogr.DataSource)
            else ogr.Open(str(datasource), gdal.GA_ReadOnly)
        )
        assert self.ds, "Could not open OGR datasource: %s" % datasource

        #: osgeo.ogr.Layer: Layer object.
        self.lyr = (
            layer
            if isinstance(layer, ogr.Layer)
            else (self.ds.GetLayerByName(layer) if layer else self.ds.GetLayer(0))
        )
        assert self.lyr, "Could not open datasource layer %s" % layer

        #: osgeo.ogr.FeatureDefn: Layer schema.
        self.schema = self.lyr.GetLayerDefn()

        #: osgeo.osr.SpatialReference: Coordinate reference system for layer.
        self.srs = self.schema.GetGeomFieldDefn(0).srs

        self._clip = False
        self._bbox_filter = None
        self._clip_geom = None
        self._iternum = 0
        logger.debug("Init FeatureReader")

    def set_bbox_filter(self, bbox, clip=False):
        """Set bbox filter for read features.

        Only features intersecting the bbox will be returned.

        Note:
            Resets reader.

        Args:
            bbox (Bbox): Bounding box. If set to `None` filter is removed.
            clip (bool, optional): Clip read features to bbox. Defaults to False.

        """
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
        logger.debug("Bbox set to %s. Clip: %s", bbox, clip)
        self.reset_reading()

    def reset_reading(self):
        """Rewinds reader to start."""
        self.lyr.ResetReading()
        logger.debug("Reset reading")

    def __iter__(self):
        """Iterate over the features."""
        self._iternum = 0
        return self

    def __next__(self):
        """Next (possibly clipped) feature which satisfies the bbox filter."""
        feat = self.lyr.GetNextFeature()
        if feat is None:
            raise StopIteration

        self._iternum += 1
        if self._iternum == 1 or self._iternum % 1000 == 0:
            logger.debug("Read %s", self._iternum)

        if self._clip:
            intersection = feat.geometry().Intersection(self._clip_geom)
            if intersection.IsEmpty():
                return self.__next__()
            feat.SetGeometryDirectly(intersection)
        return feat


def open_or_create_destination_datasource(dst_ds_name, dst_format=None, dsco=None):
    """Open an OGR DataSource for update if it exists. Otherwise create it.

    Args:
        dst_ds_name (str): DataSource string.
        dst_format (str, optional): DataSource format as specified by OGR format string. Defaults to None.
        dsco (list of str, optional): List of OGR DataSource creation options. Defaults to None.

    Raises:
        Exception: If output datasource exists, but cannot be opened in update mode.
        Exception: If `dst_format` does not identify an OGR driver.
        Exception: Datasource cannot be created.

    Returns:
        osgeo.ogr.DataSource: OGR datasource.

    """
    logger.debug("Try to open '%s' for update", dst_ds_name)
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
        logger.debug("Try to create '%s'", dst_ds_name)
        dst_ds = drv.CreateDataSource(dst_ds_name, options=dsco or [])
        if dst_ds is None:
            raise Exception("Cannot create datasource '%s'" % dst_ds_name)
    return dst_ds


def open_or_create_similar_layer(src_lyr, dst_ds, dst_lyr_name=None, lco=None):
    """Open destination vector layer if it exists. Otherwise create it as an empty copy of a source layer.

    If destination layer exists it is opened and returned. No checking is done if the schema matches the
    source layer. If the destination layer does not exist it is created with the same schema as the source layer.

    Args:
        src_lyr (osgeo.ogr.Layer): Source layer.
        dst_ds (osgeo.ogr.DataSource): Destination datasource.
        dst_lyr_name (str, optional): Destination layer name. Defaults to None.
        lco (list of str, optional): List of layer creation options as. Defaults to None.

    Raises:
        Exception: `dst_lyr_name` is None and `dst_ds` has multiple layers.
        Exception: Destination layer could not be created.

    Returns:
        osgeo.ogr.Layer: Layer.

    """
    if dst_lyr_name is None:
        count = dst_ds.GetLayerCount()
        if count > 1:
            raise Exception(
                "Destination datasource has multiple layers. Layername must be specified"
            )
        if count == 1:
            lyr = dst_ds.GetLayer(0)
        else:
            logger.debug("Creating layer 'surfclass'")
            lyr = dst_ds.CreateLayer(
                "surfclass", src_lyr.GetSpatialRef(), src_lyr.GetGeomType(), lco or []
            )
            copy_fields(src_lyr, lyr)
    else:
        lyr = dst_ds.GetLayer(dst_lyr_name)
        if lyr is None:
            logger.debug("Creating layer '%s'", dst_lyr_name)
            lyr = dst_ds.CreateLayer(
                dst_lyr_name, src_lyr.GetSpatialRef(), src_lyr.GetGeomType(), lco or []
            )
            if lyr is None:
                raise Exception('Could not create layer "%s"' % dst_lyr_name)
            copy_fields(src_lyr, lyr)
    return lyr


def copy_fields(src_lyr, dst_lyr):
    """Copy fields from `src_layer` to `dst_layer`.

    Args:
        src_lyr (osgeo.ogr.Layer): Source layer.
        dst_lyr (osgeo.ogr.Layer): Destination layer.

    Raises:
        Exception: If field cannot be created in `dst_layer`.

    """
    layer_defn = src_lyr.GetLayerDefn()
    for idx in range(layer_defn.GetFieldCount()):
        fld_defn = layer_defn.GetFieldDefn(idx)
        if dst_lyr.CreateField(fld_defn) != 0:
            raise Exception(
                'Cannot create field "%s" in layer "%s"'
                % (fld_defn.GetName(), dst_lyr.GetName())
            )
        else:
            logger.debug("Created attribute: %s", fld_defn.GetName())
