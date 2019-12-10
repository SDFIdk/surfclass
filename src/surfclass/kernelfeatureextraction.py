"""Extract feature using a kernel."""
from pathlib import Path
import numpy as np
from surfclass import rasterio, Bbox

as_strided = np.lib.stride_tricks.as_strided

# Don't allow the user to create neighborhoods larger than 13x13
# It takes longer to calculate, and removes too much information
MAX_ALLOWED_NEIGHBORHOOD = 13


class KernelFeatureExtraction:
    """Reads raster defined by bbox and extracts features using a kernel with a given neighborhood."""

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
        outputfeatures,
        neighborhood=5,
        crop_mode="reflect",
        bbox=None,
        prefix=None,
        postfix=None,
    ):
        """Create instance of KernelFeatureExtraction.

        Args:
            raster_path (str): Path to input raster.
            outdir (str): Path to output directory.
            bbox (tuple): Bounding Box of form (xmin,ymin,xmax,ymax)
            outputfeatures (list of str): list of features to extract (mean|diffmean|mean)
            neighborhood (int, optional): Size of neighborhood. Defaults to 5.
            crop_mode (str, optional): Crop mode. Defaults to "reflect".
            prefix (str, optional): Prefix to prepend output filename. Defaults to None.
            postfix (str, optional): Postfix to append output filename. Defaults to None.

        """
        # str: Path to output directory
        self.outdir = outdir or ""
        # str, "": Optional file prefix
        self.fileprefix = prefix or ""
        # str, "": Optional file postfixfix
        self.filepostfix = postfix or ""
        # rasterio.RasterReader: Class for reading rasters
        self.rasterreader = rasterio.RasterReader(raster_path)
        # surfclass.Bbox: Bounding Box (xmin,ymin,xmax,ymax)
        self.bbox = Bbox(*bbox) if bbox else self.rasterreader.bbox
        # float: Nodata value for masking
        self.nodata = self.rasterreader.nodata
        # np.array: Non-masked array read using Raster
        self.array = self.rasterreader.read_raster(bbox=self.bbox, masked=False)
        # int: Size of kernel, has to be odd
        self.neighborhood = neighborhood
        # str: Describes how to handle edges, valid options are crop or reflect
        self.crop_mode = crop_mode
        # list of str: list of features to extract (mean|diffmean|mean)
        self.outputfeatures = self._validate_feature_keys(outputfeatures)

    def _output_filename(self, feature_name):
        """Construct the output filename for the calculated tif.

        Args:
            filename (Str): basename of the output filename.

        Returns:
            Str: constructed fullpath of output file.

        """
        name = f"{self.fileprefix}{feature_name}{self.filepostfix}.tif"
        return str(Path(self.outdir) / name)

    def _validate_feature_keys(self, feat_keys):
        """Validates that the feat_keys are valid. Returns the keys if they are all valid.

        Args:
            feat_keys list::str: List of feat_keys (str)

        Returns:
            list::str: List of feat_keys (str)

        """
        accepted_keys = self.SUPPORTED_FEATURES.keys()

        # if all keys provided by user is in the accepted keys, return them.
        valid = all(map(lambda each: each in accepted_keys, feat_keys))
        assert valid, "feature is not supported, accepted features are: {}".format(
            str(accepted_keys)
        )
        return feat_keys

    def calculate_derived_features(self):
        """Calculates the neighborhood statistics for the defined raster and outputfeatures.

        Uses the matrix_as_windows function to generate a vector of length n**2 at each cell to make calculations easier.

        Yields:
            tuple(np.ma.array,str): Tuple of the calculated feature and the feature name (mean|diffmean|var)

        """
        # windows is of size (x,y,n**2) 3rd axis is the flattened n-neighborhood of that cell, including the cell itself
        windows = self.matrix_as_windows(self.array, self.neighborhood, self.crop_mode)

        if self.nodata is not None:
            mask = self.array == self.nodata
            masked_values = np.ma.masked_values(windows, self.nodata)
        else:
            mask = False
            masked_values = np.ma.array(windows)

        # Check if cropping has happened, and calculate the size of the removed edge
        if self.array.shape != masked_values[:, :, 0].shape:
            edge_size = int((self.neighborhood - 1) / 2)
        else:
            edge_size = 0

        # define indices of the inside mask of the new cropped array
        crop_indices = [
            slice(edge_size, self.array.shape[0] - edge_size),
            slice(edge_size, self.array.shape[1] - edge_size),
        ]

        for feat_name in self.outputfeatures:
            if feat_name == "mean":
                yield np.ma.masked_array(
                    np.ma.mean(masked_values, axis=2), mask=mask[tuple(crop_indices)]
                ), feat_name

            if feat_name == "diffmean":
                yield np.ma.masked_array(
                    (
                        self.array[tuple(crop_indices)]
                        - np.ma.mean(masked_values, axis=2)
                    ),
                    mask=mask[tuple(crop_indices)],
                ), feat_name

            if feat_name == "var":
                yield np.ma.masked_array(
                    np.ma.var(masked_values, axis=2), mask=mask[tuple(crop_indices)]
                ), feat_name

    @staticmethod
    def matrix_as_windows(matrix, neighborhood, crop_mode):
        """Calculate the "windows" of a x,y matrix with a given neighborhood.

        Uses np.lib.stride_tricks.as_strided to get the memory locations of the windows
        Requires matrix to be an np.array with 2 dimensions (x,y)

        Returns:
            np.array: size (x,y,neighborhood**2).

        """
        assert neighborhood % 2 == 1, "Neighborhood size has to be odd"
        assert (
            neighborhood <= MAX_ALLOWED_NEIGHBORHOOD
        ), "Neighborhood size can't be larger than {}".format(MAX_ALLOWED_NEIGHBORHOOD)

        pad_width = (neighborhood - 1) // 2

        # If crop_mode is not "crop", pad the image with the pad mode.
        # can be reflect or other modes accepted by np.pad
        if crop_mode != "crop":
            matrix = np.pad(matrix, pad_width=pad_width, mode=crop_mode)

        m_shape = matrix.shape

        # Stride magic
        # returns the indices of all cells as numpy memory locations for the neighborhood as a flattened vector like so:
        # [1 2 3
        #  4 x 6
        #  7 8 9] = [1 2 3 4 x 6 7 8 9]
        # Where x is the cell being indexed
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

    def start(self):
        """Calculate features and write to disk."""
        # Figure out the new origin based on crop_mode and neighborhood
        # If there is no crop, origin is simply UL
        if self.crop_mode == "crop":
            crop_amount = (
                int((self.neighborhood - 1) / 2) * self.rasterreader.resolution
            )
            origin = (self.bbox.xmin + crop_amount, self.bbox.ymax - crop_amount)
        else:
            origin = (self.bbox.xmin, self.bbox.ymax)

        for _, feature in enumerate(self.calculate_derived_features()):
            outfile = self._output_filename(feature[1])
            rasterio.write_to_file(
                outfile,
                feature[0],  # Array
                origin,
                self.rasterreader.resolution,
                self.rasterreader.srs,
                nodata=self.nodata,
            )
