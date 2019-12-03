import logging
import click
import numpy as np
from surfclass.scripts import options
from surfclass.vectorize import (
    FeatureReader,
    ClassCounter,
    open_or_create_destination_datasource,
    open_or_create_similar_layer,
)
from surfclass import rasterio, noise

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
def count(
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
    r"""Count occurences of cell values inside polygons.

    This tool counts the ocurrences of specified cell values whose cell center falls inside a polygon and adds the
    counts as feature attributes.

    An attribute per reported class is added. The attribute "class_n" reports the number of cells with the value n
    within the polygon and a "total_count" attribute reports the total number of cells within the polygon.

    Vector operations are carried out using the OGR library and as such datasources, layers and associated creation
    options follow the semantics specified by OGR. See https://gdal.org.

    Output dataset and output layer may exist beforehand in which case the tool tries to open it in append mode.

    Classrange defines which cell values are reported individually. Cell values not in the range will still count
    towards the total_count.

    An optional bbox can be specified to limit the features read from input datasource. Otherwise the tool will
    use the bbox of the raster file.

    If --clip is specified the geometries read from input will be clipped to the bbox used by the tool.

    Example:
    extract count --in inpolys.shp --out outpolys.geojson --format geojson --clip --classrange 0 5 classified.tif"
    """
    classes = range(classrange[0], classrange[1] + 1)

    raster_reader = rasterio.MaskedRasterReader(classraster)
    clip_bbox = bbox or raster_reader.bbox
    vector_reader = FeatureReader(indataset, inlyr)
    logger.debug("Setting bbox to: %s using clip: %s", clip_bbox, clip)
    vector_reader.set_bbox_filter(clip_bbox, clip)

    logger.debug("Creating output datasource: %s", outdataset)
    dstds = open_or_create_destination_datasource(outdataset, outformat, dsco)
    logger.debug("Creating output layer: %s", outlyr)
    dstlyr = open_or_create_similar_layer(vector_reader.lyr, dstds, outlyr, lco)

    calc = ClassCounter(vector_reader, raster_reader, dstlyr, classes)
    logger.debug("Beginning counts")
    calc.process()
    logger.debug("Done counting")


@extract.command()
@options.bbox_opt(required=False)
@click.argument("classraster", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(exists=False, dir_okay=False))
def denoise(classraster, output, bbox):
    """Applies denoising to a classified raster

    If bbox is specified only this part of the classraster will be loaded and denoised.
    If bbox is not specified the entire classraster is processed.

    CLASSRASTER is the input raster.
    The denoised output is written to OUTPUT.
    """
    logger.debug("Denoising %s", classraster)
    reader = rasterio.RasterReader(classraster)
    originX, pixel_width, _, originY, _, pixel_height = reader.geotransform
    assert np.isclose(abs(pixel_height), abs(pixel_width)), "Pixels must be square"
    logger.debug("Reading data within bbox %s", bbox)
    data = reader.read_raster(bbox=bbox, masked=True)
    logger.debug("Denoising")
    denoised = noise.denoise(data)
    logger.debug("Writing output to %s", output)
    rasterio.write_to_file(
        output, denoised, (originX, originY), abs(pixel_width), reader.srs
    )
    logger.debug("Done")
