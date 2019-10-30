import click


@click.group()
def prepare():
    """Prepare data for surfclass"""
    pass


@prepare.command()
def lidar():
    """Rasterize lidar data"""
    pass
