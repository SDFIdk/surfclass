import sys
import click
from click_plugins import with_plugins
from pkg_resources import iter_entry_points
from surfclass import __version__


def print_stderr(*args, **kwargs):
    """
    Prints messages to stderr
    The function eprint can be used in the same was as the standard print function
    :param args:
    :param kwargs:
    :return:
    """
    print(*args, file=sys.stderr, **kwargs)


@with_plugins(iter_entry_points("surfclass.surfclass_commands"))
@click.group("surfclass")
@click.version_option(version=__version__)
def cli():
    """surfclass command line interface"""
    pass


if __name__ == "__main__":
    cli()
