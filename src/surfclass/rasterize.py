"""Tools for rasterization of LiDAR files."""
import json
import logging
from pathlib import Path
import pdal
from surfclass import lidar, rasterio, Bbox

logger = logging.getLogger(__name__)

dimension_nodata = {
    "Z": -999,
    "Intensity": 0,
    "ReturnNumber": 0,
    "NumberOfReturns": 0,
    "Classification": 255,
    "ScanAngleRank": -999,
    "Pulse width": -999,
    "Amplitude": -999,
    "PointSourceId": 0,
}


class LidarRasterizer:
    """Rasterizes one or more dimensions from one or more LiDAR files.

    The underlying LiDAR library is PDAL, so dimension names used must be compatible with PDAL.

    Note:
        For the time being the following filters are hard coded into this class:
            - Ground points only (classification == 2)
            -  "Pulse width" < 2.55

    """

    def __init__(
        self,
        lidarfiles,
        outdir,
        resolution,
        bbox,
        dimensions,
        srs,
        prefix=None,
        postfix=None,
    ):
        """Inits LidarRasterizer.

        Args:
            lidarfiles (list of str): List of paths to LiDAR files.
            outdir (str): Path to output directory.
            resolution (float): Cell size in coordinate system unit.
            bbox (Bbox): Bounding box of output raster.
            dimensions (list of str): List of LiDAR dimensions to rasterize.
            srs (osgeo.osr.SpatialReference): Spatial reference system for the LiDAR files.
            prefix (str, optional): Output file(s) prefix. Defaults to None.
            postfix (str, optional): Output file(s) postfix. Defaults to None.

        """
        self.lidarfiles = (
            [lidarfiles] if isinstance(lidarfiles, (str, Path)) else list(lidarfiles)
        )
        self.outdir = outdir or ""
        self.fileprefix = prefix or ""
        self.filepostfix = postfix or ""
        self.resolution = resolution
        self.bbox = Bbox(*bbox)
        self.dimensions = self._validate_dimensions(dimensions)
        self.pipeline = self._create_pipeline()
        self.srs = srs

    def start(self):
        """Starts the processing.

        Note:
            For the time being the following filters are hard coded into this class:
                - Ground points only (classification == 2)
                -  "Pulse width" < 2.55

        Raises:
            Exception: If the PDAL pipeline built is not valid.

        """
        # Convert the pipeline to stringified JSON (required by PDAL)
        pipeline_json = json.dumps(self.pipeline)
        pipeline = pdal.Pipeline(pipeline_json)

        if pipeline.validate():
            pipeline.loglevel = 8  # really noisy
            pipeline.execute()

        else:
            logger.error("Pipeline not valid")
            raise Exception("Pipeline not valid.")

        logger.debug("Reading data")
        data = pipeline.arrays
        logger.debug("Data read: %s", data)

        # For now just assume one array
        points = data[0]

        # For now get rid of PulseWidth==2.55
        logger.warning("Dropping returns with pulsewidth >= 2.55")
        points = points[points[:]["Pulse width"] < 2.55]

        sampler = lidar.GridSampler(points, self.bbox, self.resolution)
        origin = (self.bbox.xmin, self.bbox.ymax)
        for dim in self.dimensions:
            nodata = dimension_nodata[dim]
            outfile = self._output_filename(dim)
            grid = sampler.make_grid(dim, nodata, masked=False)
            rasterio.write_to_file(
                outfile, grid, origin, self.resolution, self.srs, nodata=nodata
            )

    def _create_pipeline(self):
        # The "merge" filter is not strictly necessary according to https://pdal.io/stages/filters.merge.html#filters-merge
        # but lets be explicit about it
        pipeline = list(self.lidarfiles)

        merge = {"type": "filters.merge"}
        pipeline.append(merge)
        logger.warning("Filtering away everything but ground")
        rangefilter = {
            "type": "filters.range",
            "limits": "Classification[2:2]",  # Ground classification
        }
        pipeline.append(rangefilter)
        # xmin and ymax are inclusive, xmax and ymin are inclusive. Otherwise out gridsampler crashes
        xmin, ymin, xmax, ymax = self.bbox
        boundsfilter = {
            "type": "filters.crop",
            "bounds": f"([{xmin}, {xmax - 0.00001}], [{ymin + 0.00001}, {ymax}])",
        }
        pipeline.append(boundsfilter)

        # Build the pipeline by concating the reader, filter and writers
        return {"pipeline": pipeline}

    def _output_filename(self, dimension):
        dimname = dimension.replace(" ", "")
        name = f"{self.fileprefix}{dimname}{self.filepostfix}.tif"
        return str(Path(self.outdir) / name)

    @classmethod
    def _validate_dimensions(cls, dimensions):
        """Validates the dimensions given, against PDAL."""
        try:
            for dim in dimensions:
                if not (
                    any(
                        pdaldim["name"] == dim
                        for pdaldim in pdal.dimension.getDimensions()
                    )
                    or dim == "Pulse width"
                ):
                    raise ValueError(dim, "Dimension not recognized by PDAL")
            return dimensions
        except ValueError as e:
            print("ValueError: ", e)
