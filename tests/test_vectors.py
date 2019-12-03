from osgeo import ogr
from surfclass import Bbox
from surfclass.vectorize import (
    FeatureReader,
    ClassCounter,
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


def test_classcounter(classraster_filepath, polygons_filepath):
    vecreader = FeatureReader(polygons_filepath)
    rasreader = MaskedRasterReader(classraster_filepath)
    # Create output ds
    mem_drv = ogr.GetDriverByName("Memory")
    out_ds = mem_drv.CreateDataSource("out")
    out_lyr = open_or_create_similar_layer(vecreader.lyr, out_ds)

    in_features = list(vecreader)
    vecreader.reset_reading()

    classes = range(6)
    calc = ClassCounter(vecreader, rasreader, out_lyr, classes)
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
