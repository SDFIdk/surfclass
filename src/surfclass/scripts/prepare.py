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
    r"""Extract statistical features from a raster file

    Extract derived features from a raster file, such as mean, difference of mean and variance.

    Uses a window of size -n to calculate neighborhood statistics for each cell in the input raster.

    The output raster can either use the -c "crop" or -c "reflect" strategy to handle the edges.

    Example:
    surfclass prepare extractfeatures -b 721000 6150000 722000 6151000
        -n 5 -c reflect -f mean -f var 1km_6150_721_amplitude.tif c:\outdir\
    """
    # TODO: Add more edge handling strategies
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
    featureextractor = KernelFeatureExtraction(
        rasterfile,
        outdir,
        bbox,
        feature,
        neighborhood=neighborhood,
        crop_mode=cropmode,
        prefix=prefix,
        postfix=postfix,
    )
    logger.debug("Starting feature extraction")
    featureextractor.start()
    logger.debug("Feature extraction done!")
