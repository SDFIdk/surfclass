from pathlib import Path
import numpy as np
from surfclass.rasterreader import RasterReader
from surfclass import Bbox, rasterwriter

as_strided = np.lib.stride_tricks.as_strided

# Don't allow the user to create neighborhoods larger than 13x13
# It takes longer to calculate, and removes too much information
MAX_ALLOWED_NEIGHBORHOOD = 13


class KernelFeatureExtraction:
    """
    Reads raster defined by bbox and extracts features using a kernel with a given neighborhood.
    Can crop or "fix" the resulting rasters by padding the raster
    """

    SUPPORTED_FEATURES = {
        "mean": {
            "bandname": "mean",
            "description": "Average of neighborhood, including target cell",
        },
        "var": {
            "key": "var",
            "bandname": "var",
            "description": "Variance of neighborhood, including target cell",
        },
        "diffmean": {
            "key": "diffmean",
            "bandname": "diffmean",
            "description": "Average of neighborhood, minus target cell (cell value - mean)",
        },
    }

    def __init__(
        self,
        raster_path,
        outdir,
        bbox,
        outputfeatures,
        neighborhood=5,
        crop_mode="reflect",
        prefix=None,
        postfix=None,
    ):
        self.outdir = outdir or ""
        self.fileprefix = prefix or ""
        self.filepostfix = postfix or ""
        self.bbox = Bbox(*bbox)
        self.rasterreader = RasterReader(raster_path)
        self.nodata = self.rasterreader.nodata

        self.array = self.rasterreader.read_raster(bbox=self.bbox, masked=False)
        self.neighborhood = neighborhood
        self.crop_mode = crop_mode
        self.outputfeatures = self._validate_feature_keys(outputfeatures)

    def _output_filename(self, feature_name):
        name = f"{self.fileprefix}{feature_name}{self.filepostfix}.tif"
        return str(Path(self.outdir) / name)

    def _validate_feature_keys(self, feat_keys):
        accepted_keys = self.SUPPORTED_FEATURES.keys()

        # if all keys provided by user is in the accepted keys, return them.
        valid = all(map(lambda each: each in accepted_keys, feat_keys))
        assert valid, "feature is not supported, accepted features are: {}".format(
            str(accepted_keys)
        )

        return feat_keys

    def calculate_derived_features(self):

        features = []
        feature_names = []
        windows = self.matrix_as_windows(self.array, self.neighborhood, self.crop_mode)

        if self.nodata is not None:
            mask = self.array == self.nodata
            masked_values = np.ma.masked_values(windows, self.nodata)
        else:
            mask = False
            masked_values = np.ma.array(windows)

        # If some cropping took place
        if self.array.shape != masked_values[:, :, 0].shape:
            edge_size = int((self.neighborhood - 1) / 2)
        else:
            edge_size = 0

        crop_indices = [
            slice(edge_size, self.array.shape[0] - edge_size),
            slice(edge_size, self.array.shape[1] - edge_size),
        ]

        if "mean" in self.outputfeatures:
            features.append(
                np.ma.masked_array(
                    np.ma.mean(masked_values, axis=2), mask=mask[tuple(crop_indices)]
                )
            )
            feature_names.append("mean")

        if "diffmean" in self.outputfeatures:
            features.append(
                np.ma.masked_array(
                    (
                        self.array[tuple(crop_indices)]
                        - np.ma.mean(masked_values, axis=2)
                    ),
                    mask=mask[tuple(crop_indices)],
                )
            )
            feature_names.append("diffmean")

        if "var" in self.outputfeatures:
            features.append(
                np.ma.masked_array(
                    np.ma.var(masked_values, axis=2), mask=mask[tuple(crop_indices)]
                )
            )
            feature_names.append("var")

        return (features, feature_names)

    def start(self):
        """
        Calculates the derived features, mean and variance for a given neighborhood and crop_mode
        Returns list of numpy arrays
        """
        # Read raw values out of raster, as matrix_as_windows does not worked on masked arrays

        features, feature_names = self.calculate_derived_features()

        # Figure out the new origin based on crop_mode and neighborhood
        # If there is no crop, origin is simply UL
        if self.crop_mode == "crop":
            #
            crop_amount = (
                int((self.neighborhood - 1) / 2) * self.rasterreader.resolution
            )
            origin = (self.bbox.xmin + crop_amount, self.bbox.ymax - crop_amount)
        else:
            origin = (self.bbox.xmin, self.bbox.ymax)

        for idx, feature in enumerate(features):
            outfile = self._output_filename(feature_names[idx])
            rasterwriter.write_to_file(
                outfile,
                feature,
                origin,
                self.rasterreader.resolution,
                25832,
                nodata=self.nodata,
            )

    @staticmethod
    def matrix_as_windows(matrix, neighborhood, crop_mode):
        """
        Calculates the "windows" of a x,y matrix with a given neighborhood
        Uses np.lib.stride_tricks.as_strided to get the memory locations of the windows
        Requires matrix to be an np.array with 2 dimensions (x,y)
        Returns: np.array of size (x,y,neighborhood**2).
        """
        assert neighborhood % 2 == 1, "Neighborhood size has to be odd"
        assert (
            neighborhood <= MAX_ALLOWED_NEIGHBORHOOD
        ), "Neighborhood size can't be larger than {}".format(MAX_ALLOWED_NEIGHBORHOOD)

        pad_width = (neighborhood - 1) // 2

        # If crop_mode is not "crop", pad the image with the crop mode.
        # can be reflect or other modes accepted by np.pad
        if crop_mode != "crop":
            matrix = np.pad(matrix, pad_width=pad_width, mode=crop_mode)

        m_shape = matrix.shape

        matrix_windows = as_strided(
            matrix,
            shape=(
                matrix.shape[0] - neighborhood + 1,
                matrix.shape[1] - neighborhood + 1,
                neighborhood,
                neighborhood,
            ),
            strides=(
                matrix.strides[0],
                matrix.strides[1],
                matrix.strides[0],
                matrix.strides[1],
            ),
            writeable=False,  # Use this to avoid writing to memory in weird places
        )

        return matrix_windows.reshape(
            m_shape[0] - neighborhood + 1,
            m_shape[1] - neighborhood + 1,
            neighborhood ** 2,
        )
