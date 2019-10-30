import logging
import click

logger = logging.getLogger(__name__)


@click.group()
def prepare():
    """Prepare data for surfclass"""


@prepare.command()
def lidar():
    """Rasterize lidar data"""
    logger.debug("Preparing lidar data")
