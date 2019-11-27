from osgeo import gdal
import numpy as np

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
        # TODO: replace this with a MaskedRasterReader
        # TODO: add output path
        # TODO: Handle naming of output file _var, _mean etc.
        self._ds = gdal.Open(str(raster_path), gdal.GA_ReadOnly)
        assert self._ds, "Could not open raster"

        self._bbox = bbox
        self.geotransform = self._ds.GetGeoTransform()
        self._band = self._ds.GetRasterBand(1)
        self.nodata = self._band.GetNoDataValue()
        self.array = self.readBandAsArray()

    def readBandAsArray(self):
        src_offset = self.bbox_to_pixel_window()
        return self._band.ReadAsArray(*src_offset)

    def bbox_to_pixel_window(self):
        xmin, ymin, xmax, ymax = self._bbox
        originX, pixel_width, _, originY, _, pixel_height = self.geotransform
        x1 = int((xmin - originX) / pixel_width)
        x2 = int((xmax - originX) / pixel_width + 0.5)
        y1 = int((ymax - originY) / pixel_height)
        y2 = int((ymin - originY) / pixel_height + 0.5)
        xsize = x2 - x1
        ysize = y2 - y1
        return (x1, y1, xsize, ysize)

    def calculate_derived_features(self, neighborhood=5, crop_mode=True):
        features = []
        # band_names = []

        windows = self.matrix_as_windows(self.array, neighborhood, crop_mode)

        if self.nodata is not None:
            mask = np.ma.masked_values(windows, self.nodata)
        else:
            mask = np.ma.array(windows)

        # TODO: add argument to determine what features we want here.
        features.append(np.ma.mean(mask, axis=2))
        features.append(np.ma.var(mask, axis=2))

        # band_names.append(band_name + "_mean")
        # band_names.append(band_name + "_var")

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
