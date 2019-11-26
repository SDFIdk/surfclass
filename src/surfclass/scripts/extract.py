import logging
import click
from surfclass.scripts import options
from surfclass.vectorize import (
    MaskedRasterReader,
    FeatureReader,
    StatsCalculator,
    open_or_create_destination_datasource,
    open_or_create_similar_layer,
)

logger = logging.getLogger(__name__)


@click.group()
def extract():
    """Extract data from surfclass classified raster"""


@extract.command()
@click.option(
    "--in", "indataset", default=None, required=True, help="Input OGR dataset"
)
@click.option("--inlyr", default=None, required=False, help="Layer in input dataset")
@click.option(
    "--out", "outdataset", default=None, required=True, help="Output OGR dataset"
)
@click.option("--outlyr", default=None, required=False, help="Layer in output dataset")
@click.option(
    "--format", "outformat", default=None, required=False, help="Output OGR format"
)
@click.option("--clip/--no-clip", default=False, help="Clip input features to bbox")
@options.bbox_opt(required=False)
@click.option(
    "--classrange", type=(int, int), required=True, help="Classes to collect (min, max)"
)
@click.option(
    "--dsco",
    type=str,
    required=False,
    multiple=True,
    help="OGR dataset creation option",
)
@click.option(
    "--lco", type=str, required=False, multiple=True, help="OGR layer creation option"
)
@click.argument("classraster", type=str)
def stats(
    indataset,
    inlyr,
    outdataset,
    outlyr,
    outformat,
    clip,
    bbox,
    classrange,
    dsco,
    lco,
    classraster,
):
    classes = range(classrange[0], classrange[1] + 1)

    raster_reader = MaskedRasterReader(classraster)
    clip_bbox = bbox or raster_reader.bbox
    vector_reader = FeatureReader(indataset, inlyr)
    logger.debug("Setting bbox to: %s using clip: %s", clip_bbox, clip)
    vector_reader.set_bbox_filter(clip_bbox, clip)

    logger.debug("Creating output datasource: %s", outdataset)
    dstds = open_or_create_destination_datasource(outdataset, outformat, dsco)
    logger.debug("Creating output layer: %s", outlyr)
    dstlyr = open_or_create_similar_layer(vector_reader.lyr, dstds, outlyr, lco)

    calc = StatsCalculator(vector_reader, raster_reader, dstlyr, classes)
    logger.debug("Beginning stats calculations")
    calc.process()
    logger.debug("Done stats calculations")
