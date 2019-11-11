# pylint: disable=W0613
import click
from surfclass import Bbox

verbosity_arg = click.option(
    "--verbosity",
    "-v",
    type=click.Choice(
        ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"], case_sensitive=False
    ),
    default="ERROR",
    help="Set verbosity level",
)


def bbox_handler(ctx, param, value):
    if not value or (isinstance(value, tuple) and not all(value)):
        return None
    return Bbox(*value)


def bbox_opt(required=False):
    return click.option(
        "-b",
        "--bbox",
        required=required,
        default=[None] * 4,
        callback=bbox_handler,
        type=(float, float, float, float),
        help="BBOX: xmin ymin xmax ymax",
    )


def resolution_handler(ctx, param, value):
    retval = float(value)
    if retval <= 0:
        raise ValueError("Resolution must be greater than zero")
    return retval


resolution_opt = click.option(
    "-r",
    "--resolution",
    type=float,
    required=True,
    callback=resolution_handler,
    help="Grid cell size",
)
