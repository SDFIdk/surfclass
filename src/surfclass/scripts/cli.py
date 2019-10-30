import sys
import click
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


@click.group("surfclass")
@click.version_option(version=__version__)
def cli():
    pass


if __name__ == "__main__":
    cli()
