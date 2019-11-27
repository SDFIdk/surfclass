import numpy as np
from surfclass.rasterreader import RasterReader

as_strided = np.lib.stride_tricks.as_strided

# Don't allow the user to create neighborhoods larger than 13x13
# It takes longer to calculate, and removes too much information
MAX_ALLOWED_NEIGHBORHOOD = 13


class KernelFeatureExtraction:
    """
    Reads raster defined by bbox and extracts features using a kernel with a given neighborhood.
    Can crop or "fix" the resulting rasters by padding the raster
    """

    def __init__(self, raster_path, bbox):

        self.rasterreader = RasterReader(raster_path)
        self._bbox = bbox
        self.nodata = self.rasterreader.nodata
        self.array = self.rasterreader.read_raster(bbox=self._bbox, masked=False)

    def calculate_derived_features(self, neighborhood=5, crop_mode=True):
        """
        Calculates the derived features, mean and variance for a given neighborhood and crop_mode
        Returns list of numpy arrays
        """
        features = []

        windows = self.matrix_as_windows(self.array, neighborhood, crop_mode)

        if self.nodata is not None:
            mask = np.ma.masked_values(windows, self.nodata)
        else:
            mask = np.ma.array(windows)

        # TODO: add argument to determine what features we want here.
        features.append(np.ma.mean(mask, axis=2))
        features.append(np.ma.var(mask, axis=2))

        return features

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
