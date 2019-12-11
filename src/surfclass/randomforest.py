"""Classification using random forest."""
import logging
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class RandomForest:
    """Train or classify using a RandomForest model."""

    def __init__(self, num_features, num_trees=200, model=None):
        """Create instance of RandomForest.

        Args:
            num_features (int): Number of features to train or classify.
            num_trees (int, optional): [description]. Defaults to 200.
            model ([type], optional): [description]. Defaults to None.

        """
        self.num_features = num_features
        self.num_trees = num_trees
        self.model = self.load_model(model)

    def load_model(self, model):
        """Load trained sklearn.ensemble.RandomForestClassifier model.

        Args:
            model_path (str): path to the trained model

        Returns:
            sklearn.ensemble.RandomForestClassifier: Trained model, see reference for details.

        """
        if model is None:
            return None
        # Check if the model_input is a path or an sklearn random forest model
        if isinstance(model, str):
            try:
                model = pickle.load(open(model, "rb"))
                return self.validate_model(model)
            except OSError:
                logger.error("Could not load RandomForestModel")
                return None

        elif isinstance(model, RandomForestClassifier):
            # Validate model based on parameters
            return self.validate_model(model)

        return None

    def validate_model(self, model):
        """Validate a model with the current class instantiation.

        Args:
            model (sklearn.ensemble.RandomForestClassifier): A trained RandomForestClassifier

        Returns:
            [sklearn.ensemble.RandomForestClassifier]: A valid trained RandomForestClassifier

        """
        if not isinstance(model, RandomForestClassifier):
            logger.error(
                "Can not validate model, is not of instance sklearn.ensemble.forest.RandomForestClassifier"
            )
            return None

        if not model.n_estimators == self.num_trees:
            logger.error(
                "Number of trees is different from model parameter. Model has: %d, input was: %d",
                model.n_estimators,
                self.num_trees,
            )
            return None

        if not model.n_features_ == self.num_features:
            logger.error(
                "Number of features is different from model parameter. Model has: %d, input was: %d",
                model.n_features_,
                self.num_features,
            )
            return None

        return model

    def train(self, X, y):
        """Train/Fit a RandomForestClassifier using the observation matrix X and class vector y.

        Args:
            X (np.array): 2D Matrix of feature observations.
            y (np.array): 1D vector of class labels.h

        Returns:
            sklearn.ensemble.RandomForestClassifier: A trained RandomForestClassifier model.

        """
        # If a model is already defined, something is wrong. Does not support training multiple times in a row.
        if self.model is not None:
            logger.error(
                "Surfclass does not support training an already existing model.."
            )
            return None

        # validate X fits the parameters given in init
        assert isinstance(X, np.ndarray), "X is not a valid numpy.ndarray"
        assert (
            X.ndim == 2
        ), "X does not have the correct shape, should be of form (n,f): observations 1D, and feature"
        assert y.ndim == 1, "y does not have the correct shape, should be 1D vector"
        assert (
            X.shape[1] == self.num_features
        ), "Model and input does have the same number of features"
        assert (
            X.shape[0] == y.shape[0]
        ), "Number of class observations does not match number of feature observations."

        rf = RandomForestClassifier(
            n_estimators=self.num_trees, oob_score=False, verbose=0, n_jobs=-1
        )

        # fit the model
        rf_trained = rf.fit(X, y)

        # save the model to the instanced class (useful when one want to run classify immediately after)
        self.model = rf_trained

        # return the trained model
        return rf_trained

    def classify(self, X):
        """Classify X using the instantiated RandomForestClassifier model.

        Args:
            X (np.array): 2D Matrix of feature observations.

        Returns:
            np.array: classified vector.

        """
        assert (
            self.model is not None
        ), "Could not find a model, please either train a model or initialise the class with a valid model path"

        # TODO: This might be double-work but the model attribute can have been changed
        model = self.validate_model(self.model)

        # Test the X input is acceptable for the given model.
        assert (
            X.ndim == 2
        ), "X does not have the correct shape, should be of form (n,f): observations 1D, and feature"
        assert isinstance(X, np.ndarray), "X is not a valid numpy array"
        assert (
            X.shape[1] == self.num_features
        ), "Model and input does have the same number of features"

        # run the classificaiton using X
        class_prediction = model.predict(X)

        return class_prediction
