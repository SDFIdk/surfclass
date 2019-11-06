import os
import json
import logging
import pdal
from surfclass import lidar, rasterwriter, Bbox

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
}


class LidarRasterizer:
    def __init__(self, resolution, bbox, lidarfile, dimensions, filterexp):
        self.resolution = resolution
        self.bbox = Bbox(*bbox)
        self.lidar = lidarfile
        self.dimensions = self._validate_dimensions(dimensions)
        self.reader = self._create_pipeline_reader(lidarfile)
        self.filterexp = filterexp

    def start(self):
        xmin, ymin, xmax, ymax = self.bbox
        logger.warning("Filtering away everything but ground")
        rangefilter = {
            "type": "filters.range",
            "limits": "Classification[2:2]",  # Ground classification
        }
        # xmin and ymax are inclusive, xmax and ymin are inclusive. Otherwise out gridsampler crashes
        boundsfilter = {
            "type": "filters.crop",
            "bounds": f"([{xmin}, {xmax - 0.00001}], [{ymin + 0.00001}, {ymax}])",
        }

        # Build the pipeline by concating the reader, filter and writers
        pipeline_dict = {"pipeline": [self.reader, boundsfilter, rangefilter]}
        pipeline_json = json.dumps(pipeline_dict)
        logger.debug("Using pipeline: %s", pipeline_json)

        # Convert the pipeline to stringified JSON (required by PDAL)
        pipeline = pdal.Pipeline(pipeline_json)

        if pipeline.validate():
            pipeline.loglevel = 8  # really noisy
            pipeline.execute()

        else:
            logger.error("Pipeline not valid")

        logger.debug("Reading data")
        data = pipeline.arrays
        logger.debug("Data read: %s", data)

        # For now just assume one array
        points = data[0]

        # For now get rid of PulseWidth==2.55
        logger.warning("Dropping returns with pulsewidth >= 2.55")
        points = points[points[:]["Pulse width"] < 2.55]

        sampler = lidar.GridSampler(points, self.bbox, self.resolution)
        origin = (xmin, ymax)

        for dim in self.dimensions:
            nodata = dimension_nodata[dim]
            outfile = self._output_filename(dim)
            grid = sampler.make_grid(dim, nodata, masked=False)
            rasterwriter.write_to_file(
                outfile, grid, origin, self.resolution, 25832, nodata=nodata
            )

    @classmethod
    def _create_pipeline_reader(cls, lidarfile):
        return {"type": "readers.las", "filename": lidarfile}

    @classmethod
    def _output_filename(cls, dimension):
        return "1km_6184_720_terrain_" + dimension.replace(" ", "") + ".tif"

    @classmethod
    def _validate_dimensions(cls, dimensions):
        """Validates the dimensions given, against PDAL"""
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


def test():
    """ Only used for internal testing """
    resolution = 0.5  # Coarse resolution for fast testing
    bbox = Bbox(727000, 6171000, 728000, 6172000)
    lidarfile = os.path.join(
        "/Volumes/GoogleDrive/My Drive/Septima - Ikke synkroniseret/Projekter/SDFE/Befæstelse/data/trænings_las",
        "1km_6171_727.las",
    )
    dimensions = ["Intensity", "Amplitude", "Pulse width"]
    r = LidarRasterizer(resolution, bbox, lidarfile, dimensions, "")

    r.start()


if __name__ == "__main__":
    test()
