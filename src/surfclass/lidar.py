import logging
import pdal
from pdal.pipeline import Pipeline
import numpy as np

logger = logging.getLogger(__name__)


def open_pdal_pipeline(lidar_file):
    pipeline_str = '{"pipeline": ["%s"]}' % lidar_file
    pipeline = pdal.Pipeline(pipeline_str)
    pipeline.validate()  # check if our JSON and options were good
    pipeline.loglevel = 8
    return pipeline


def read_into_numpy(pdal_pipeline):
    if not isinstance(pdal_pipeline, Pipeline):
        raise TypeError("pdal_pipeline must be of type 'pdal.pipeline.Pipeline'")
    if not pdal_pipeline.arrays:
        logger.debug("Running pipeline.execute()")
        count = pdal_pipeline.execute()
        logger.debug("Pipeline returned: %d", count)

    logger.info("Pipeline metadata: \n %s", pdal_pipeline.metadata)
    return pdal_pipeline.arrays


class GridSampler:
    def __init__(self, lidar_points, bbox, resolution):
        self.points = lidar_points
        self.bbox = bbox
        self.resolution = resolution
        self.grid_shape = self._grid_shape()
        self.cell_indexes = self._cell_indexes()

    def crop_to_bbox(self):
        """Removes points falling outside the given bbox"""
        xmin, ymin, xmax, ymax = self.bbox
        maskx = np.logical_and(self.points["X"] >= xmin, self.points[:]["X"] < xmax)
        masky = np.logical_and(self.points["Y"] > ymin, self.points[:]["Y"] <= ymax)
        mask = np.logical_and(maskx, masky)
        self.points = self.points[mask]
        self.cell_indexes = self._cell_indexes()

    def _cell_indexes(self):
        # cell row indexes
        xmin, _, _, ymax = self.bbox
        col_ixes = ((self.points[:]["X"] - xmin) / self.resolution).astype(int)
        # cell col indexes
        row_ixes = ((self.points[:]["Y"] - ymax) / (-1 * self.resolution)).astype(int)
        return (row_ixes, col_ixes)

    def _grid_shape(self):
        xmin, ymin, xmax, ymax = self.bbox
        dx, dy = np.abs(xmax - xmin), np.abs(ymax - ymin)
        rows, cols = (
            np.int(np.ceil(dy / self.resolution)),
            np.int(np.ceil(dx / self.resolution)),
        )
        return (rows, cols)

    def make_grid(self, dimension, nodata=0, masked=True):
        if not isinstance(dimension, str):
            raise TypeError("dimension must be a string")

        if not dimension in self.points.dtype.fields:
            valid_fields = self.points.dtype.fields
            raise ValueError(
                f"dimension '{dimension}' not found in data ({valid_fields})"
            )

        datatype = self.points.dtype[dimension]
        if not np.can_cast(nodata, datatype):
            raise TypeError(
                f"nodata value {nodata} cannot be cast to dimension dtype {datatype}"
            )
        logger.debug(
            "Creating %s '%s' grid. Nodata: '%s'. Masked: '%s' ",
            self.grid_shape,
            datatype,
            nodata,
            masked,
        )
        out_grid = np.ones(self.grid_shape, datatype) * nodata
        logger.info("Gridding dimension %s", dimension)
        out_grid[self.cell_indexes[0], self.cell_indexes[1]] = self.points[:][dimension]
        if not masked:
            return out_grid
        logger.debug("Masking")
        mask = out_grid == nodata
        return np.ma.array(out_grid, mask=mask)
