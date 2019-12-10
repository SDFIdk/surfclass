import logging
import pathlib
import click
from scipy import stats
from surfclass.scripts import options
from surfclass.rasterize import LidarRasterizer
from surfclass.kernelfeatureextraction import KernelFeatureExtraction
from surfclass import train

logger = logging.getLogger(__name__)


@click.group()
def prepare():
    """Prepare data for surfclass."""


@prepare.command()
@options.bbox_opt(required=True)
@options.srs_opt
@options.resolution_opt
@click.option(
    "-d",
    "--dimension",
    type=str,
    multiple=True,
    required=True,
    help="lidar dimension to rasterize. As defined by PDAL. Multiple allowed.",
)
@click.option("--prefix", default=None, required=False, help="Output file prefix")
@click.option("--postfix", default=None, required=False, help="Output file postfix")
@click.argument(
    "lidarfile",
    type=click.Path(exists=True, dir_okay=False),
    # Allow multiple input files to be given
    nargs=-1,
)
@click.argument("outdir", type=click.Path(exists=False, file_okay=False), nargs=1)
def lidargrid(lidarfile, bbox, srs, resolution, dimension, outdir, prefix, postfix):
    r"""Rasterize lidar data

    Rasterize one or more lidar files into grid cells.

    Example:

    surfclass prepare lidargrid -srs epsg:25832 -b 721000 6150000 722000 6151000 -r 0.4
        -d Intensity -d Z 1km_6150_721.las 1km_6149_720.las c:\outdir\

    """
    # Log inputs
    logger.debug(
        "lidargrids started with arguments: %s, %s, %s, %s, %s, %s, %s, %s",
        lidarfile,
        bbox,
        srs.ExportToPrettyWkt(),
        resolution,
        dimension,
        outdir,
        prefix,
        postfix,
    )

    # Make sure output dir exists
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    rizer = LidarRasterizer(
        lidarfile,
        outdir,
        resolution,
        bbox,
        dimension,
        srs,
        prefix=prefix,
        postfix=postfix,
    )
    logger.debug("Starting rasterisation")
    rizer.start()
    logger.debug("Rasterisation ended")


@prepare.command()
@options.bbox_opt(required=False)
@click.option(
    "-n",
    "--neighborhood",
    type=int,
    multiple=False,
    required=True,
    help="Size of neighborhood kernel. Has to be an odd number.",
)
@click.option(
    "-c",
    "--cropmode",
    type=str,
    multiple=False,
    required=True,
    help="How to handle cropping, accepted valued are: (crop|reflect)",
)
@click.option("--prefix", default=None, required=False, help="Output file prefix")
@click.option("--postfix", default=None, required=False, help="Output file postfix")
@click.argument(
    "rasterfile",
    type=click.Path(exists=True, dir_okay=False),
    # Allow only one argument
    # TODO: extend class to take multiple files at the same time
    nargs=1,
)
@click.option(
    "-f",
    "--feature",
    type=click.Choice(KernelFeatureExtraction.SUPPORTED_FEATURES.keys()),
    multiple=True,
    required=True,
    help="Feature to extract. Multiple allowed.",
)
@click.argument("outdir", type=click.Path(exists=False, file_okay=False), nargs=1)
def extractfeatures(
    rasterfile, bbox, neighborhood, feature, cropmode, outdir, prefix, postfix
):
    r"""Extract statistical features from a raster file.

    Extract derived features from a raster file, such as mean, difference of mean and variance.
    Uses a window of size -n to calculate neighborhood statistics for each cell in the input raster.

    The output raster can either use the -c "crop" or -c "reflect" strategy to handle the edges.
    "crop" removes a surrounding edge of size (n-1)/2 from the array.
    "reflect" pads the array with an edge of size (n-1)/2 by "reflecting"/"mirroring" the data at the edge.

    The bbox is used when *reading* the raster. If the strategy is "crop" the resulting bbox will be smaller.

    Example:
<<<<<<< HEAD
    surfclass prepare extractfeatures -b 721000 6150000 722000 6151000
        -n 5 -c reflect -f mean -f var 1km_6150_721_amplitude.tif c:\outdir\
=======
        surfclass prepare extractfeatures -b 721000 6150000 722000 6151000
            -n 5 -c reflect -f mean -f var 1km_6150_721_amplitude.tif c:\outdir\
>>>>>>> safer geotransform comparison and cleanup

    """
    # Log inputs
    logger.debug(
        "extractfeatures started with arguments: %s, %s, %s, %s, %s,%s, %s, %s",
        rasterfile,
        bbox,
        neighborhood,
        feature,
        cropmode,
        outdir,
        prefix,
        postfix,
    )

    # Make sure output dir exists
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    # Initialize the KernelFeatureExtraction class
    featureextractor = KernelFeatureExtraction(
        rasterfile,
        outdir,
        feature,
        bbox=bbox,
        neighborhood=neighborhood,
        crop_mode=cropmode,
        prefix=prefix,
        postfix=postfix,
    )
    logger.debug("Starting feature extraction")
    featureextractor.start()
    logger.debug("Feature extraction done!")


@prepare.command()
@click.option(
    "--in",
    "indataset",
    default=None,
    required=True,
    help="OGR dataset with draining polygons",
)
@click.option("--inlyr", default=None, required=False, help="Layer name")
@click.option(
    "--attrib",
    "-a",
    default=None,
    required=True,
    help="Name of attribute defining class (as Integer)",
)
@click.option(
    "-f",
    "--feature",
    "rasterfiles",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    multiple=True,
    required=True,
    help="Feature raster file. Multiple allowed. NOTE: Order is important!!!",
)
@click.argument("outputfile", type=click.Path(exists=False, file_okay=True), nargs=1)
def traindata(indataset, inlyr, attrib, rasterfiles, outputfile):
    """Extracts training data defined by polygons with a class from a set of raster features.

    Example:
    surfclass prepare traindata --in train_polys.gpkg --inlyr areas --attrib classno -f feature1.tif
        -f feature2.tif -f feature3.tif my_traning_data.npz

    """
    # Print feature order. This is important.
    click.echo("Extracting training data from features:")
    for i, fp in enumerate(rasterfiles):
        click.echo(f"f{i+1}: {fp}")

    (classes, features) = train.collect_training_data(
        indataset, inlyr, attrib, rasterfiles
    )
    click.echo("Stats for extracted training data:")
    click.echo(stats.describe(classes))
    click.echo("Stats for extracted feature data:")
    click.echo(stats.describe(features))
    train.save_training_data(outputfile, rasterfiles, classes, features)


@prepare.command()
@click.argument("datafile", type=click.Path(exists=True, file_okay=True), nargs=1)
def traindatainfo(datafile):
    """Shows basic information about extracted training data.

    Example:
    surclass prepare traindatainfo my_traning_data

    """
    # TODO: Beautify output like
    # Number of observations: xxx
    # f1: min=x max=y mean=z
    # f2: ...
    file_paths, classes, features = train.load_training_data(datafile)
    click.echo("Trained from features:")
    for i, fp in enumerate(file_paths):
        click.echo(f"f{i+1}: {fp}")
    click.echo("Stats for classes:")
    click.echo(stats.describe(classes))
    click.echo("Stats for features:")
    click.echo(stats.describe(features))
