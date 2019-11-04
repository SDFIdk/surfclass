import logging
import click
from click_plugins import with_plugins
from pkg_resources import iter_entry_points
from surfclass import __version__
from surfclass.scripts import options
from surfclass.scripts.helpers import ClickColoredLoggingFormatter, ClickLoggingHandler

logger = logging.getLogger(__name__)


def configure_logging(log_level):
    handler = ClickLoggingHandler()
    handler.formatter = ClickColoredLoggingFormatter("%(name)s: %(message)s")
    logging.basicConfig(level=log_level.upper(), handlers=[handler])


@with_plugins(iter_entry_points("surfclass.surfclass_commands"))
@click.group("surfclass")
@click.version_option(version=__version__)
@options.verbosity_arg
def cli(verbosity):
    """surfclass command line interface"""
    configure_logging(verbosity)
