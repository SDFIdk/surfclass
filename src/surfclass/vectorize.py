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


class ClassCounter:
    def __init__(self, raster_path):
        self._ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
        assert self._ds, "Could not open raster"
        self._band = self._ds.GetRasterBand(1)
        self._bbox = None
        self.geotransform = self._ds.GetGeoTransform()
        self.nodata = self._band.GetNoDataValue()

        # Memory drivers
        self._ogr_mem_drv = ogr.GetDriverByName("Memory")
        self._gdal_mem_drv = gdal.GetDriverByName("MEM")

    @property
    def bbox(self):
        if self._bbox is None:
            xmin = self.geotransform[0]
            ymax = self.geotransform[3]
            width = self.geotransform[1] * self._ds.RasterXSize
            height = self.geotransform[5] * self._ds.RasterYSize
            xmax = xmin + width
            ymin = ymax + height
            return Bbox(xmin, ymin, xmax, ymax)

    def bbox_to_pixel_window(self, bbox):
        originX, pixel_width, _, originY, _, pixel_height = self.geotransform
        x1 = int((bbox[0] - originX) / pixel_width)
        x2 = int((bbox[1] - originX) / pixel_width) + 1
        y1 = int((bbox[3] - originY) / pixel_height)
        y2 = int((bbox[2] - originY) / pixel_height) + 1
        xsize = x2 - x1
        ysize = y2 - y1
        return (x1, y1, xsize, ysize)

    def count_classes_inside(self, geom):
        if not isinstance(geom, ogr.Geometry):
            raise TypeError("Must be OGR geometry")
        mem_type = geom.GetGeometryType()
        src_offset = self.bbox_to_pixel_window(geom.GetEnvelope())
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
        mem_layer = mem_ds.CreateLayer("mem_lyr", None, mem_type)
        mem_feature = ogr.Feature(mem_layer.GetLayerDefn())
        mem_feature.SetGeometry(geom)
        mem_layer.CreateFeature(mem_feature)

        # Rasterize the feature
        mem_raster_ds = self._gdal_mem_drv.Create(
            "", src_offset[2], src_offset[3], 1, gdal.GDT_Byte
        )
        mem_raster_ds.SetGeoTransform(new_gt)
        # Burn 1 inside our feature
        gdal.RasterizeLayer(mem_raster_ds, [1], mem_layer, burn_values=[1])
        rasterized_array = mem_raster_ds.ReadAsArray()

        # Mask the source data array with our current feature mask
        masked = np.ma.MaskedArray(src_array, mask=np.logical_not(rasterized_array))

        # Ok, now count classes (including nodata):
        unique, counts = np.unique(masked.compressed(), return_counts=True)
        class_counts = dict(zip(unique, counts))
        return class_counts


class FeatureReader:
    def __init__(self, datasource_string, layer=None):
        self.ds = ogr.Open(datasource_string, gdal.GA_ReadOnly)
        assert self.ds, "Could not open OGR datasource"
        self.lyr = self.ds.GetLayerByName(layer) if layer else self.ds.GetLayer(0)
        assert self.lyr, "Could not open datasource layer"
        self.schema = self.lyr.GetLayerDefn()
        self._clip = False
        self._bbox_filter = None
        self._clip_geom = None

    def set_bbox_filter(self, bbox, clip=False):
        self._clip = clip
        self._bbox_filter = Bbox(*bbox) if bbox else None
        if clip and bbox:
            self._clip_geom = bbox_to_ogr_polygon(bbox)
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


def test():
    class_raster = "/Users/asger/Downloads/1km_6171_727_scanangle_intensity_amplitude_prediction/1km_6171_727_scanangle_intensity_amplitude_prediction.tif"
    poly_ds = "/Volumes/GoogleDrive/My Drive/Septima - Ikke synkroniseret/Projekter/SDFE/Befæstelse/data/trænings_polygoner/1km_6171_727.sqlite"
    poly_lyr = "1km_6171_727"

    counter = ClassCounter(class_raster)
    raster_bbox = counter.bbox
    reader = FeatureReader(poly_ds, poly_lyr)
    reader.set_bbox_filter(raster_bbox, clip=True)
    for f in reader:
        classes = counter.count_classes_inside(f.geometry())
        print(classes)


if __name__ == "__main__":
    test()
