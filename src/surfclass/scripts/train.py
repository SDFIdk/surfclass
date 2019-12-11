import logging
import pathlib
import pickle
import click
from scipy import stats
from surfclass.randomforest import RandomForest
from surfclass.train import load_training_data

logger = logging.getLogger(__name__)


@click.group()
def train():
    """Train surface classification models using path to training_data."""


@train.command()
@click.option(
    "-n",
    "--numtrees",
    type=int,
    multiple=False,
    required=False,
    default=100,
    help="Number of trees/estimators, has to follow n > num_features (default is 100)",
)
@click.argument("trainingdata", type=click.Path(exists=True, file_okay=True), nargs=1)
@click.argument("outputfile", type=click.Path(exists=False, file_okay=True), nargs=1)
def testmodel1(trainingdata, outputfile, numtrees):
    r"""Trains a new testmodel1 model using an .npz file generated by "surfclass prepare traindata [OPTIONS].

    The traindata should match the model definition.

    Example: surfclass train testmodel1 "testmodel1.npz" "testmodel1.sav"

    """
    (_, classes, features) = load_training_data(trainingdata)

    # This might be a good idea to expose to the CLI.
    # Log inputs
    logger.debug(
        "Training testmodel1 with arguments: %s, %s,%s",
        trainingdata,
        outputfile,
        numtrees,
    )

    classifier = RandomForest(4, model=None)
    logger.debug("Training testmodel1...")
    rf_trained = classifier.train(features, classes, num_trees=numtrees)

    pickle.dump(rf_trained, open(outputfile, "wb"))
    logger.debug(
        "Training done, written .sav to: %s", pathlib.Path(outputfile).resolve()
    )


@train.command()
@click.option(
    "-n",
    "--numtrees",
    type=int,
    multiple=False,
    required=False,
    default=100,
    help="Number of trees/estimators",
)
@click.argument("trainingdata", type=click.Path(exists=True, file_okay=True), nargs=1)
@click.argument("outputfile", type=click.Path(exists=False, file_okay=True), nargs=1)
def randomforestndvi(trainingdata, outputfile, numtrees):
    r"""Trains a new randomforestndvi model using an .npz file generated by "surfclass prepare traindata [OPTIONS].

    The traindata should match the model definition.

    Example:
        surfclass train randomforestndvi "randomforestndvi.npz" "randomforestndvi.sav"

    """
    (_, classes, features) = load_training_data(trainingdata)

    # Log inputs
    logger.debug(
        "Training randomforestndvi with arguments: %s, %s, %s",
        trainingdata,
        outputfile,
        numtrees,
    )
    # TODO: Might make sense to generalize the train function, if so pass in features.shape[1] instead of "10"
    classifier = RandomForest(10, model=None)
    logger.debug("Training randomforestndvi")
    rf_trained = classifier.train(features, classes, num_trees=numtrees)

    pickle.dump(rf_trained, open(outputfile, "wb"))
    logger.debug(
        "Training done, written .sav to: %s", pathlib.Path(outputfile).resolve()
    )


@train.command()
@click.option(
    "-n",
    "--numtrees",
    type=int,
    multiple=False,
    required=False,
    default=100,
    help="Number of trees/estimators",
)
@click.argument("trainingdata", type=click.Path(exists=True, file_okay=True), nargs=1)
@click.argument("outputfile", type=click.Path(exists=False, file_okay=True), nargs=1)
def genericmodel(trainingdata, outputfile, numtrees):
    r"""Trains a new randomforestndvi model using an .npz file generated by "surfclass prepare traindata [OPTIONS].

    The traindata should match the model definition.

    Example:
        surfclass train randomforestndvi "randomforestndvi.npz" "randomforestndvi.sav"

    """
    (_, classes, features) = load_training_data(trainingdata)

    click.echo("Stats for feature data:")
    click.echo(stats.describe(features))

    # Log inputs
    logger.debug(
        "Training randomforestndvi with arguments: %s, %s, %s",
        trainingdata,
        outputfile,
        numtrees,
    )

    classifier = RandomForest(features.shape[1], model=None)
    logger.debug("Training Model...")
    rf_trained = classifier.train(features, classes, num_trees=numtrees)

    pickle.dump(rf_trained, open(outputfile, "wb"))
    logger.debug(
        "Training done, written .sav to: %s", pathlib.Path(outputfile).resolve()
    )
