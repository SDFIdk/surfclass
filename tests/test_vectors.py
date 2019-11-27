from osgeo import ogr, osr
import numpy as np
from surfclass import Bbox
from surfclass.vectorize import (
    FeatureReader,
    StatsCalculator,
    open_or_create_similar_layer,
)
from surfclass.rasterreader import MaskedRasterReader


def test_featurereader(polygons_filepath):
    reader = FeatureReader(polygons_filepath, None)
    assert reader
    assert isinstance(reader.schema, ogr.FeatureDefn)
    features = list(reader)
    assert len(features) == 83
    # Set bbox filter. No clip
    reader.set_bbox_filter(Bbox(727420.4, 6171605.6, 727500.2, 6171683.0))
    features = list(reader)
    assert len(features) == 26
    area_orig = sum([x.geometry().Area() for x in features])
    assert area_orig == 19162.50219995297
    # Set bbox filter. Clip features to bbox
    reader.set_bbox_filter(Bbox(727420.4, 6171605.6, 727500.2, 6171683.0), clip=True)
    features = list(reader)
    assert len(features) == 26
    area_clipped = sum([x.geometry().Area() for x in features])
    assert area_clipped == 3783.6933658819635
    # Reset filter
    reader.set_bbox_filter(None)
    features = list(reader)
    assert len(features) == 83


def test_maskedrasterreader(classraster_filepath):
    reader = MaskedRasterReader(classraster_filepath)
    assert reader
    assert isinstance(reader.srs, osr.SpatialReference)
    assert reader.bbox == Bbox(727000.0, 6171000.0, 728000.0, 6172000.0)
    pix_win = reader.bbox_to_pixel_window(reader.bbox)
    assert pix_win == (0, 0, 500, 500)
    pix_win = reader.bbox_to_pixel_window(
        Bbox(727500.0, 6171600.0, 727600.0, 6171750.0)
    )
    assert pix_win == (250, 125, 50, 75)
    # Try using a point
    pnt = ogr.Geometry(ogr.wkbPoint)
    pnt.AddPoint(727500.0, 6171600.0)
    pnt.AssignSpatialReference(reader.srs)
    data = reader.read_masked(pnt)
    np.testing.assert_array_equal(data, np.ma.empty(shape=(0, 0)))
    # Now use a proper polygon
    poly = pnt.Buffer(100)
    data = reader.read_masked(poly)
    assert data.shape == (100, 100)
    # Check that we have a mask
    assert int(np.sum(data.mask)) == 2140
    # Check data without mask
    assert int(np.sum(data.data)) == 25412
    # Check data with mask. (Must be less than unmasked)
    assert int(np.sum(data.compressed())) == 20432


def test_statscalculator(classraster_filepath, polygons_filepath):
    vecreader = FeatureReader(polygons_filepath)
    rasreader = MaskedRasterReader(classraster_filepath)
    # Create output ds
    mem_drv = ogr.GetDriverByName("Memory")
    out_ds = mem_drv.CreateDataSource("out")
    out_lyr = open_or_create_similar_layer(vecreader.lyr, out_ds)

    in_features = list(vecreader)
    vecreader.reset_reading()

    classes = range(6)
    calc = StatsCalculator(vecreader, rasreader, out_lyr, classes)
    calc.process()

    out_reader = FeatureReader(out_ds, out_lyr)
    out_reader.reset_reading()

    out_features = list(out_reader)
    assert len(out_features) == len(in_features)
    out_fields = set(out_features[0].keys())
    expected_classes = ["class_%s" % x for x in classes]
    expected = expected_classes + ["id", "total_count"]
    assert out_fields == set(expected)

    for inf, outf in zip(in_features, out_features):
        assert inf["id"] == outf["id"]
        assert inf.geometry().ExportToWkt() == outf.geometry().ExportToWkt()
        total = sum([outf[x] for x in expected_classes])
        assert total == outf["total_count"]

    # Check a couple features
    values = [out_features[3][x] for x in expected_classes]
    assert values == [0, 1, 0, 15, 3, 0]
    values = [out_features[59][x] for x in expected_classes]
    assert values == [151, 3, 0, 1, 18, 0]
