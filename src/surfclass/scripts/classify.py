import logging
import pathlib
import click
import numpy as np
from surfclass.scripts import options
from surfclass.randomforest import RandomForest
from surfclass.classify import stack_rasters
from surfclass.rasterio import write_to_file

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
    r""" TEST:
    Create a surface classified raster using a set of input features and a trained RandomForest model.

    The input features must match exactly as described and in the correct order.

    Example surfclass classify randomforestndvi -b 721000 6150000 722000 6151000 -f1 1km_6150_721_Amplitude.tif
    -f2 1km_6150_721_Amplitude_diffmean.tif -f3 1km_6150_721_Amplitude_mean.tif -f4 1km_6150_721_Amplitude_var.tif
    testmodel1.sav c:\outdir\
    """
    # Log inputs
    logger.debug(
        "Classification with testmodel1 started with arguments: %s, %s, %s, %s, %s, %s, %s, %s",
        feature1,
        feature2,
        feature3,
        feature4,
        model,
        outdir,
        prefix,
        postfix,
    )
    filename = "classification"
    fileprefix = prefix or ""
    filepostfix = postfix or ""
    # Make sure output dir exists
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    # Read the input rasters and stack them into an np.ndarray
    features = [feature1, feature2, feature3, feature4]
    (X, mask, geotransform, srs, _shape) = stack_rasters(features, bbox)
    indices = np.where(mask)[0]
    # Instantiate the RandomForest model
    classifier = RandomForest(len(features), model=model)

    # Classify X using the instantiated RandomForest model
    logger.debug("Starting classification")
    class_prediction = classifier.classify(X)
    logger.debug("Finished classification")

    output = np.zeros(mask.shape[0])

    # Convert to byte array to save space
    output[indices] = np.int8(class_prediction)
    output = output.reshape(_shape)
    name = f"{fileprefix}{filename}{filepostfix}.tif"
    outpath = str(pathlib.Path(outdir) / name)

    # Get origin and resolution from geotransform
    origin = (geotransform[0], geotransform[3])
    resolution = geotransform[1]
    logger.debug("Writing classification output here: %s", outpath)
    write_to_file(outpath, output, origin, resolution, srs, nodata=0)


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
    r"""
    RandomForestNDVI

    Create a surface classified raster using a set of input features and a trained RandomForest model.

    The input features must match exactly as described and in the correct order.

    Example:  surfclass classify randomforestndvi -b 721000 6150000 722000 6151000 -f1 1km_6150_721_Amplitude.tif
                                                                     -f2 1km_6150_721_Amplitude_mean.tif
                                                                     -f3 1km_6150_721_Amplitude_var.tif
                                                                     -f4 1km_6150_721_NDVI.tif
                                                                     -f5 1km_6150_721_NDVI_mean.tif
                                                                     -f6 1km_6150_721_NDVI_var.tif
                                                                     -f7 1km_6150_721_Pulsewidth.tif
                                                                     -f8 1km_6150_721_Pulsewidth_mean.tif
                                                                     -f9 1km_6150_721_Pulsewidth_var.tif
                                                                     -f10 1km_6171_727_ReturnNumber.tif
                                                                     A5_NDVI5_P5_R_NT400_1km_6171_727_40cm_predicted.sav
                                                                     c:\outdir\
    """
    # Log inputs
    logger.debug(
        "Classification with randomforestndvi started with arguments: %s, %s, %s, %s, %s, %s, %s,%s,%s, %s, %s, %s, %s, %s",
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

    filename = "classification"
    fileprefix = prefix or ""
    filepostfix = postfix or ""
    # Make sure output dir exists
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    # Read the input rasters and stack them into an np.ndarray
    features = [
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
    ]
    (X, mask, geotransform, srs, _shape) = stack_rasters(features, bbox)
    indices = np.where(mask)[0]
    # Instantiate the RandomForest model
    classifier = RandomForest(len(features), model=model)

    # Classify X using the instantiated RandomForest model
    logger.debug("Starting classification")
    class_prediction = classifier.classify(X)
    logger.debug("Finished classification")

    output = np.zeros(mask.shape[0])

    # Convert to byte array to save space
    output[indices] = np.int8(class_prediction)
    output = output.reshape(_shape)
    name = f"{fileprefix}{filename}{filepostfix}.tif"
    outpath = str(pathlib.Path(outdir) / name)

    # Get origin and resolution from geotransform
    origin = (geotransform[0], geotransform[3])
    resolution = geotransform[1]
    logger.debug("Writing classification output here: %s", outpath)
    write_to_file(outpath, output, origin, resolution, srs, nodata=0)


@classify.command()
@options.bbox_opt(required=True)
@click.option(
    "-f",
    "--feature",
    "rasterfiles",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    multiple=True,
    required=True,
    help="Feature raster file. Multiple allowed. NOTE: Order is important!!!",
)
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
def genericmodel(rasterfiles, model, outdir, bbox, prefix, postfix):
    r"""
    Generic Model

    Create a surface classified raster using a set of input features and a trained RandomForest model.

    The input features must match the model provided. Add "-v INFO" to check that the order of the input
    rasters meet expectations

    Example:  surfclass classify genericmodel -b 721000 6150000 722000 6151000
                                                                     -f 1km_6150_721_Amplitude.tif
                                                                     -f 1km_6150_721_Amplitude_mean.tif
                                                                     -f 1km_6150_721_Amplitude_var.tif
                                                                     -f 1km_6150_721_NDVI.tif
                                                                     -f 1km_6150_721_NDVI_mean.tif
                                                                     -f 1km_6150_721_NDVI_var.tif
                                                                     -f 1km_6150_721_Pulsewidth.tif
                                                                     -f 1km_6150_721_Pulsewidth_mean.tif
                                                                     -f 1km_6150_721_Pulsewidth_var.tif
                                                                     -f 1km_6171_727_ReturnNumber.tif
                                                                     genericmodel.sav
                                                                     c:\outdir\
    """
    # Log inputs
    logger.debug(
        "Classification with model %s started with arguments: %s, %s, %s",
        model,
        outdir,
        prefix,
        postfix,
    )

    # Print feature order. This is important.
    click.echo("Classifying using %s with features:" % model)
    for i, fp in enumerate(rasterfiles):
        click.echo(f"f{i+1}: {fp}")

    filename = "classification"
    fileprefix = prefix or ""
    filepostfix = postfix or ""
    # Make sure output dir exists
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    # Read the input rasters and stack them into an np.ndarray
    features = rasterfiles

    (X, mask, geotransform, srs, _shape) = stack_rasters(features, bbox)
    indices = np.where(mask)[0]
    # Instantiate the RandomForest model
    classifier = RandomForest(len(features), model=model)

    # Classify X using the instantiated RandomForest model
    logger.debug("Starting classification")
    class_prediction = classifier.classify(X)
    logger.debug("Finished classification")

    output = np.zeros(mask.shape[0])

    # Convert to byte array to save space
    output[indices] = np.int8(class_prediction)
    output = output.reshape(_shape)
    name = f"{fileprefix}{filename}{filepostfix}.tif"
    outpath = str(pathlib.Path(outdir) / name)

    # Get origin and resolution from geotransform
    origin = (geotransform[0], geotransform[3])
    resolution = geotransform[1]
    logger.debug("Writing classification output here: %s", outpath)
    write_to_file(outpath, output, origin, resolution, srs, nodata=0)
