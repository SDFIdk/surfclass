import click
import sys


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
@click.version_option(version="0.0.1")
def cli():
    pass


if __name__ == "__main__":
    cli()
