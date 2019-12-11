import numpy as np
from surfclass import Bbox
from surfclass.kernelfeatureextraction import KernelFeatureExtraction


def test_kernelfeatureextraction(amplituderaster_filepath, tmp_path):
    bbox = Bbox(727000, 6171000, 728000, 6172000)

    extractor = KernelFeatureExtraction(
        amplituderaster_filepath,
        tmp_path,
        ["mean", "var", "diffmean"],
        bbox=bbox,
        prefix="test",
        crop_mode="crop",
    )

    # Test basic assertion about the input array
    assert extractor
    assert extractor.array.shape == (250, 250)
    assert extractor.array.sum() == -4189377.63
    assert extractor.nodata == -999.0

    # Calculate derived features with the "crop" method
    feature_generator = extractor.calculate_derived_features()
    derived_features, _ = zip(*list(feature_generator))

    assert len(derived_features) == 3
    # Since we cropped size is smaller then input.
    assert derived_features[0].shape == (246, 246)

    # Calculate derived features with the "reflect" method
    extractor.crop_mode = "reflect"
    # Get a new generator
    feature_generator = extractor.calculate_derived_features()
    derived_features, _ = zip(*list(feature_generator))
    assert len(derived_features) == 3
    # Since we reflected output shape is equal to input shape
    assert derived_features[0].shape == (250, 250)

    # Test that mean and variance calculation in "simple cases" are correct
    # Area is picked such that no "nodata" cells are included
    assert extractor.array[110:115, 110:115].mean() == derived_features[0][112, 112]
    assert extractor.array[110:115, 110:115].var() == derived_features[1][112, 112]

    # Test DiffMean is correct
    diffmean = extractor.array[112, 112] - extractor.array[110:115, 110:115].mean()
    assert diffmean == derived_features[2][112, 112]


def test_kernelfeatureextraction_nobbox(amplituderaster_filepath, tmp_path):

    extractor = KernelFeatureExtraction(
        amplituderaster_filepath,
        tmp_path,
        ["mean", "var", "diffmean"],
        prefix="test",
        crop_mode="crop",
    )

    # Test basic assertion about the input array
    assert extractor
    assert extractor.array.shape == (250, 250)
    assert extractor.array.sum() == -4189377.63
    assert extractor.nodata == -999.0

    # Calculate derived features with the "crop" method
    feature_generator = extractor.calculate_derived_features()
    derived_features, _ = zip(*list(feature_generator))

    assert len(derived_features) == 3
    # Since we cropped size is smaller then input.
    assert derived_features[0].shape == (246, 246)

    # Calculate derived features with the "reflect" method
    extractor.crop_mode = "reflect"
    # Get a new generator
    feature_generator = extractor.calculate_derived_features()
    derived_features, _ = zip(*list(feature_generator))
    assert len(derived_features) == 3
    # Since we reflected output shape is equal to input shape
    assert derived_features[0].shape == (250, 250)

    assert all(x[0].dtype == "float32" for x in derived_features)

    # Test that mean and variance calculation in "simple cases" are correct
    # Area is picked such that no "nodata" cells are included
    assert (
        np.float32(extractor.array[110:115, 110:115].mean())
        == derived_features[0][112, 112]
    )
    assert (
        np.float32(extractor.array[110:115, 110:115].var())
        == derived_features[1][112, 112]
    )

    # Test DiffMean is correct
    diffmean = extractor.array[112, 112] - extractor.array[110:115, 110:115].mean()
    assert np.float32(diffmean) == derived_features[2][112, 112]
