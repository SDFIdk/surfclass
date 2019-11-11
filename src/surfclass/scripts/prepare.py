import logging
import pathlib
import click
from surfclass.scripts import options
from surfclass.rasterize import LidarRasterizer

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
