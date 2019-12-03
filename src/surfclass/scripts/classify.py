import logging
import pathlib
import click
from surfclass.scripts import options
from surfclass.randomforestclassifier import RandomForestClassifier

logger = logging.getLogger(__name__)


@click.group()
def classify():
    """Surface classify raster"""


@classify.command()
@options.bbox_opt(required=True)
@click.option("-f1", "--feature1", required=True, help="Amplitude")
@click.option("-f2", "--feature2", required=True, help="Diffmean Amplitude n=3")
@click.option("-f3", "--feature3", required=True, help="Mean Amplitude n=3")
@click.option("-f4", "--feature4", required=True, help="Var Amplitude n=3")
@click.option("--prefix", default=None, required=False, help="Output file prefix")
@click.option("--postfix", default=None, required=False, help="Output file postfix")
@click.argument("model", type=click.Path(exists=True, dir_okay=False))
@click.argument("outdir", type=click.Path(exists=False, file_okay=False), nargs=1)
def testmodel1(
    feature1, feature2, feature3, feature4, model, outdir, bbox, prefix, postfix
):
    # Log inputs
    logger.debug(
        "Classification with testmodel1 started with arguments: %s, %s, %s, %s, %s, %s, %s,%s",
        feature1,
        feature2,
        feature3,
        feature4,
        model,
        outdir,
        prefix,
        postfix,
    )
    # Make sure output dir exists
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    classifier = RandomForestClassifier(
        model,
        [feature1, feature2, feature3, feature4],
        bbox,
        outdir,
        prefix=prefix,
        postfix=postfix,
    )
    logger.debug("Starting classification")
    classifier.start()
    logger.debug("Classification done, written to: %s", outdir)


@classify.command()
@options.bbox_opt(required=True)
@click.option("-f1", "--feature1", required=True, help="Amplitude")
@click.option("-f2", "--feature2", required=True, help="Amplitude Mean n=5")
@click.option("-f3", "--feature3", required=True, help="Amplitude Var n=5")
@click.option("-f4", "--feature4", required=True, help="NDVI")
@click.option("-f5", "--feature5", required=True, help="NDVI Mean n=5")
@click.option("-f6", "--feature6", required=True, help="NDVI Var n=5")
@click.option("-f7", "--feature7", required=True, help="Pulse width n=5")
@click.option("-f8", "--feature8", required=True, help="Pulse width Mean n=5")
@click.option("-f9", "--feature9", required=True, help="Pulse width Var n=5")
@click.option("-f10", "--feature10", required=True, help="ReturnNumber")
@click.option("--prefix", default=None, required=False, help="Output file prefix")
@click.option("--postfix", default=None, required=False, help="Output file postfix")
@click.argument(
    "model",
    type=click.Path(exists=True, dir_okay=False),
    # Allow just one model
    nargs=1,
)
@click.argument(
    "outdir",
    type=click.Path(exists=False, file_okay=False),
    # Allow just one output directory
    nargs=1,
)
def randomforestndvi(
    feature1,
    feature2,
    feature3,
    feature4,
    feature5,
    feature6,
    feature7,
    feature8,
    feature9,
    feature10,
    model,
    outdir,
    bbox,
    prefix,
    postfix,
):
    # Log inputs
    logger.debug(
        "Classification with testmodel1 started with arguments: %s, %s, %s, %s, %s, %s, %s,%s,%s, %s, %s, %s, %s, %s",
        feature1,
        feature2,
        feature3,
        feature4,
        feature5,
        feature6,
        feature7,
        feature8,
        feature9,
        feature10,
        model,
        outdir,
        prefix,
        postfix,
    )
    # Make sure output dir exists
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    classifier = RandomForestClassifier(
        model,
        [
            feature1,
            feature2,
            feature3,
            feature4,
            feature5,
            feature6,
            feature7,
            feature8,
            feature9,
            feature10,
        ],
        bbox,
        outdir,
        prefix=prefix,
        postfix=postfix,
    )
    logger.debug("Starting classification")
    classifier.start()
    logger.debug("Classification done, written to: %s", outdir)
