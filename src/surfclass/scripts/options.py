import click

verbosity_arg = click.option(
    "--verbosity",
    "-v",
    type=click.Choice(
        ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"], case_sensitive=False
    ),
    default="ERROR",
    help="Set verbosity level",
)
