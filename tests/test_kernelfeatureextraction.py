from surfclass import Bbox
from surfclass.kernelfeatureextraction import KernelFeatureExtraction


def test_kernelfeatureextraction(amplituderaster_filepath, tmp_path):
    bbox = Bbox(727000, 6171000, 728000, 6172000)

    extractor = KernelFeatureExtraction(
        amplituderaster_filepath, tmp_path, bbox, prefix="test", crop_mode="crop"
    )

    # Test basic assertion about the input array
    assert extractor
    assert extractor.array.shape == (250, 250)
    assert extractor.array.sum() == -4189377.63
    assert extractor.nodata == -999.0

    # Calculate derived features with the "crop" method
    derived_features, _ = extractor.calculate_derived_features()
    assert len(derived_features) == 2
    # Since we cropped size is smaller then input.
    assert derived_features[0].shape == (246, 246)

    # Calculate derived features with the "reflect" method
    extractor.crop_mode = "reflect"
    derived_features, _ = extractor.calculate_derived_features()
    assert len(derived_features) == 2
    # Since we reflected output shape is equal to input shape
    assert derived_features[0].shape == (250, 250)

    # Test that mean and variance calculation in "simple cases" are correct
    # Area is picked such that no "nodata" cells are included
    # assert extractor.array[110:115, 110:115].mean() == derived_features[0][112, 112]
    # assert extractor.array[110:115, 110:115].var() == derived_features[1][112, 112]
