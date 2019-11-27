import logging
import pathlib
import click
from surfclass.scripts import options
from surfclass.rasterize import LidarRasterizer
from surfclass.kernelfeatureextraction import KernelFeatureExtraction

logger = logging.getLogger(__name__)


@click.group()
def prepare():
    """Prepare data for surfclass"""


@prepare.command()
@options.bbox_opt(required=True)
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
def lidargrid(lidarfile, bbox, resolution, dimension, outdir, prefix, postfix):
    r"""Rasterize lidar data

    Rasterize one or more lidar files into grid cells.

    Example:

    surfclass prepare lidargrid -b 721000 6150000 722000 6151000 -r 0.4
        -d Intensity -d Z 1km_6150_721.las 1km_6149_720.las c:\outdir\

    """
    # Log inputs
    logger.debug(
        "lidargrids started with arguments: %s, %s, %s, %s, %s, %s, %s",
        lidarfile,
        bbox,
        resolution,
        dimension,
        outdir,
        prefix,
        postfix,
    )

    # Make sure output dir exists
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    rizer = LidarRasterizer(
        lidarfile, outdir, resolution, bbox, dimension, prefix=prefix, postfix=postfix
    )
    logger.debug("Starting rasterisation")
    rizer.start()
    logger.debug("Rasterisation ended")


@prepare.command()
@options.bbox_opt(required=True)
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
    help="How to handle cropping, accepted valued are: crop or reflect",
)
@click.argument(
    "rasterfile",
    type=click.Path(exists=True, dir_okay=False),
    # Allow only one argument
    # TODO: extend class to take multiple files at the same time
    nargs=1,
)
@click.argument("outdir", type=click.Path(exists=False, file_okay=False), nargs=1)
def extractfeatures(rasterfile, bbox, neighborhood, cropmode, outdir):
    r"""Extract features from a raster file

    Extract derived features from a raster file, such as mean or variance

    Example:

    surfclass prepare extractfeatures -b 721000 6150000 722000 6151000
        -n 5 -c reflect 1km_6150_721_amplitude.tif c:\outdir\

    """
    # Log inputs
    logger.debug(
        "extractfeatures started with arguments: %s, %s, %s, %s, %s",
        rasterfile,
        bbox,
        neighborhood,
        cropmode,
        outdir,
    )

    # Make sure output dir exists
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    featureextractor = KernelFeatureExtraction(rasterfile, bbox)
    logger.debug("Starting feature extraction")
    features = featureextractor.calculate_derived_features(
        neighborhood=neighborhood, crop_mode=cropmode
    )
    logger.debug("Feature extraction done!")
    logger.debug("Writing features to disk..")
