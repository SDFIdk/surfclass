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
    def __init__(
        self,
        lidarfiles,
        outdir,
        resolution,
        bbox,
        dimensions,
        prefix=None,
        postfix=None,
        filterexp=None,
    ):
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
        self.filterexp = filterexp

    def start(self):
        # Convert the pipeline to stringified JSON (required by PDAL)
        pipeline_json = json.dumps(self.pipeline)
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
        origin = (self.bbox.xmin, self.bbox.ymax)
        for dim in self.dimensions:
            nodata = dimension_nodata[dim]
            outfile = self._output_filename(dim)
            grid = sampler.make_grid(dim, nodata, masked=False)
            rasterio.write_to_file(
                outfile, grid, origin, self.resolution, 25832, nodata=nodata
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

    kvnetixes = [(6167, 729), (6171, 727), (6176, 724), (6184, 720), (6220, 717)]

    for kvnetix in kvnetixes:
        bbox = Bbox(
            kvnetix[1] * 1000,
            kvnetix[0] * 1000,
            kvnetix[1] * 1000 + 1000,
            kvnetix[0] * 1000 + 1000,
        )
        tile = f"1km_{kvnetix[0]}_{kvnetix[1]}"
        lidarfile = (
            Path(
                "/Volumes/GoogleDrive/My Drive/Septima - Ikke synkroniseret/Projekter/SDFE/Befæstelse/data/trænings_las"
            )
            / f"{tile}.las"
        )
        outdir = ""
        prefix = f"{tile}_"
        dimensions = [
            "Intensity",
            "Amplitude",
            "Pulse width",
            "ReturnNumber",
            "ScanAngleRank",
            "PointSourceId",
        ]
        r = LidarRasterizer(
            str(lidarfile), outdir, resolution, bbox, dimensions, prefix=prefix
        )
        r.start()


if __name__ == "__main__":
    test()
