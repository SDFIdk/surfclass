from pathlib import Path
import logging
import pickle
import numpy as np
from surfclass import Bbox, rasterwriter, rasterreader


logger = logging.getLogger(__name__)


class RandomForestClassifier:
    """
     Performs a RandomForestClassification using an input stack of raster files,
     and a pickled trained sklearn.ensemble.RandomForestClassifier model.
    """

    def __init__(self, model_path, features, bbox, outdir, prefix=None, postfix=None):
        self.model = self._load_model(
            model_path[0]
        )  # TODO: Why is the path parameter as a tuple here?
        self.feature_paths = features
        self.outdir = outdir or ""
        self.fileprefix = prefix or ""
        self.filepostfix = postfix or ""
        self.bbox = Bbox(*bbox)
        self.datastack = None
        self.resolution = 0.00001

    @staticmethod
    def _load_model(model_path):
        """loads trained sklearn.ensemble.RandomForestClassifier model
        Args:
            model_path (Str): path to the trained model
        Returns:
            sklearn.ensemble.RandomForestClassifier: Trained model, see reference for details.
        """
        # assert Path(
        #    model_path
        # ).is_file(), "Model file was not found on location {}".format(model_path)

        try:
            model = pickle.load(open(model_path, "rb"))
            return model
        except OSError:
            logger.error("Could not load RandomForestModel")
            return None

    def _output_filename(self, filename):
        """Constructs the output filename for the predicted tif
        Args:
            filename (Str): basename of the output filename
        Returns:
            Str: constructed fullpath of output file
        """
        name = f"{self.fileprefix}{filename}{self.filepostfix}.tif"
        return str(Path(self.outdir) / name)

    def stack_features(self):
        """Stacks the features provided to the class along the 3rd axis
        Returns:
            np.ma.ndarray: Masked 3D ndarray in the form (x,y,n) where n is the raster band
        """
        features = []
        for f in self.feature_paths:
            rr = rasterreader.RasterReader(f)
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
        # Not sure the np.ma.dstack() is doing what we're expecting here
        stacked_features = np.ma.dstack(features)

        return stacked_features

    def start(self):

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
        outfile = self._output_filename("prediction")
        rasterwriter.write_to_file(
            outfile,
            class_prediction.filled(fill_value=0),
            (self.bbox.xmin, self.bbox.ymax),
            self.resolution,
            25832,
            nodata=0,
        )
