from surfclass import Bbox
from surfclass.kernelfeatureextraction import KernelFeatureExtraction


def test_kernelfeatureextraction(amplituderaster_filepath):
    bbox = Bbox(727000, 6171000, 728000, 6172000)

    extractor = KernelFeatureExtraction(amplituderaster_filepath, bbox)
    assert extractor

    # Calculate derived features with the "crop" method
    derived_features = extractor.calculate_derived_features(
        neighborhood=5, crop_mode="crop"
    )
    assert len(derived_features) == 2
    # Since we cropped size is smaller then input.
    assert derived_features[0].shape == (246, 246)

    # Calculate derived features with the "reflect" method
    derived_features = extractor.calculate_derived_features(
        neighborhood=5, crop_mode="reflect"
    )
    assert len(derived_features) == 2
    # Since we reflected output shape is equal to input shape
    assert derived_features[0].shape == (250, 250)
