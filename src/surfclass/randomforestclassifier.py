from pathlib import Path
import logging
import pickle
import numpy as np
from surfclass import Bbox, rasterio


logger = logging.getLogger(__name__)


class RandomForestClassifier:
    """Performs a RandomForest Classification by reading multiple raster inputs and a trained model."""

    def __init__(self, model_path, features, bbox, outdir, prefix=None, postfix=None):
        """Create instance of RandomForestClassifier.

        Args:
            model_path (str): Path to the trained sklearn.RandomForest model ex. model.sav
            features (list of str): List of paths for the raster features used. Sorted by entry in model
            bbox (tuple): Bounding Box of form (xmin,ymin,xmax,ymax)
            outdir (str): Path to output directory
            prefix (str, optional): Prefix to prepend output filename. Defaults to None.
            postfix (str, optional): Postfix to append output filename. Defaults to None.

        """
        # sklearn.ensemble.RandomForestClassifier: Random Forest Model.
        self.model = self._load_model(model_path)
        # list of str: Paths to input rasters
        self.feature_paths = features
        # str: Path to output directory
        self.outdir = outdir or ""
        # str, "": Optional file prefix
        self.fileprefix = prefix or ""
        # str, "": Optional file postfixfix
        self.filepostfix = postfix or ""
        # surfclass.Bbox: Bounding Box (xmin,ymin,xmax,ymax)
        self.bbox = Bbox(*bbox)
        # np.ma.array: Stacked array of features
        self.datastack = None
        # float: Resolution used when writing to raster.
        self.resolution = 0.00001

    @staticmethod
    def _load_model(model_path):
        """Load trained sklearn.ensemble.RandomForestClassifier model.

        Args:
            model_path (str): path to the trained model

        Returns:
            sklearn.ensemble.RandomForestClassifier: Trained model, see reference for details.

        """
        try:
            model = pickle.load(open(model_path, "rb"))
            return model
        except OSError:
            logger.error("Could not load RandomForestModel")
            return None

    def _output_filename(self, filename):
        """Construct the output filename for the predicted tif.

        Args:
            filename (Str): basename of the output filename

        Returns:
            Str: constructed fullpath of output file

        """
        name = f"{self.fileprefix}{filename}{self.filepostfix}.tif"
        return str(Path(self.outdir) / name)

    def stack_features(self):
        """Stack features provided to the class along the 3rd axis.

        Returns:
            np.ma.ndarray: Masked 3D ndarray in the form (x,y,n) where n is the raster band

        """
        features = []
        for f in self.feature_paths:
            rr = rasterio.RasterReader(f)
            nodata = rr.nodata
            # TODO: Replace this with check of equal geotransforms
            self.resolution = max(self.resolution, rr.resolution)
            array = rr.read_raster(bbox=self.bbox, masked=False)
            # TODO: Continuing issue. Come up with common way to treat this
            if nodata is not None:
                array = np.ma.masked_values(array, nodata)
            else:
                array = np.ma.array(array)

            features.append(array)
        # Stack the features along the 3rd axis.
        stacked_features = np.ma.dstack(features)

        return stacked_features

    def start(self):
        """Calculates the classification array and writes the resulting raster to disk.

        Uses the input bbox, resolution and output folder to write the raster to disk.
        """
        X = self.stack_features()
        rf = self.model

        # Flatten datastack
        X_flatten = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
        # Predict using model
        class_prediction = rf.predict(X_flatten)
        # Reshape prediction to 2D
        class_prediction = class_prediction.reshape(X.shape[0], X.shape[1])

        # TODO: Figure out a way to "look" through the datastack and find the common mask
        # This solution is much too slow
        # mask_or = np.ma.array(
        #    [np.ma.mask_or(X.mask[:, :, 0], mask) for mask in X.mask[:, :, 1:].ravel()]
        # )

        # Hack:
        # Take the mask of the first feature.
        mask_or = X.mask[:, :, 0]

        class_prediction = np.ma.masked_array(class_prediction, mask=mask_or)

        # Write the output to disk
        outfile = self._output_filename("classification")
        rasterio.write_to_file(
            outfile,
            class_prediction.filled(fill_value=0),
            (self.bbox.xmin, self.bbox.ymax),
            self.resolution,
            25832,
            nodata=0,
        )
