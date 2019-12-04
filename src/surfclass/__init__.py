"""Process LiDAR data into a surface clasified raster."""
from collections import namedtuple

__version__ = "0.0.1"
__description__ = "Processes Lidar-Data into a surface classified raster"
__author__ = "Asger Skovbo Petersen"
__email__ = "asger@septima.dk"
__uri__ = "https://github.com/septima/surfclass"
__license__ = "Licensed under the MIT license"

Bbox = namedtuple("Bbox", ["xmin", "ymin", "xmax", "ymax"])
