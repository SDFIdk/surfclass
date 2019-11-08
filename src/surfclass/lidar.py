import logging
import json
import pdal
from pdal.pipeline import Pipeline
import numpy as np

logger = logging.getLogger(__name__)


def open_pdal_pipeline(lidar_file):
    pipeline_obj = {"pipeline": [str(lidar_file)]}
    pipeline_str = json.dumps(pipeline_obj)
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
        self.use_min_scanangle = True
        self._points = lidar_points
        self._bbox = bbox
        self._resolution = resolution
        self._grid_shape = self._calc_grid_shape()
        self._prepared = False
        self._cell_indexes = None

    def crop_to_bbox(self):
        """Removes points falling outside the given bbox"""
        xmin, ymin, xmax, ymax = self._bbox
        maskx = np.logical_and(self._points["X"] >= xmin, self._points[:]["X"] < xmax)
        masky = np.logical_and(self._points["Y"] > ymin, self._points[:]["Y"] <= ymax)
        mask = np.logical_and(maskx, masky)
        self._points = self._points[mask]
        self._prepared = False

    def _prepare(self):
        if self.use_min_scanangle:
            # Order points by descending abs(scananglerank)
            # This eventually gives us the echo with the smallest abs(scananglerank)) for each output cell
            abs_angle = np.abs(self._points[:]["ScanAngleRank"])
            # the indices that would sort the abs_angle array
            sorted_ix_abs_angle = np.argsort(abs_angle)
            # Sort points by desc abs_angle
            self._points = self._points[sorted_ix_abs_angle[::-1]]

        # Now calculate cell indexes for the sorted points
        self._cell_indexes = self._calc_cell_indexes()

    def _calc_cell_indexes(self):
        # cell row indexes
        xmin, _, _, ymax = self._bbox
        col_ixes = ((self._points[:]["X"] - xmin) / self._resolution).astype(int)
        # cell col indexes
        row_ixes = ((self._points[:]["Y"] - ymax) / (-1 * self._resolution)).astype(int)
        return (row_ixes, col_ixes)

    def _calc_grid_shape(self):
        xmin, ymin, xmax, ymax = self._bbox
        dx, dy = np.abs(xmax - xmin), np.abs(ymax - ymin)
        rows, cols = (
            np.int(np.ceil(dy / self._resolution)),
            np.int(np.ceil(dx / self._resolution)),
        )
        return (rows, cols)

    def make_grid(self, dimension, nodata=0, masked=True):
        if not self._prepared:
            self._prepare()
        if not isinstance(dimension, str):
            raise TypeError("dimension must be a string")

        if not dimension in self._points.dtype.fields:
            valid_fields = self._points.dtype.fields
            raise ValueError(
                f"dimension '{dimension}' not found in data ({valid_fields})"
            )

        datatype = self._points.dtype[dimension]
        if not np.can_cast(nodata, datatype):
            raise TypeError(
                f"nodata value {nodata} cannot be cast to dimension dtype {datatype}"
            )
        logger.debug(
            "Creating %s '%s' grid. Nodata: '%s'. Masked: '%s' ",
            self._grid_shape,
            datatype,
            nodata,
            masked,
        )
        out_grid = np.ones(self._grid_shape, datatype) * nodata
        logger.info("Gridding dimension %s", dimension)
        out_grid[self._cell_indexes[0], self._cell_indexes[1]] = self._points[:][
            dimension
        ]
        if not masked:
            return out_grid
        logger.debug("Masking")
        mask = out_grid == nodata
        return np.ma.array(out_grid, mask=mask)
