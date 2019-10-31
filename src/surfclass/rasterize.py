import os
import json
import logging
import pprint
import pdal

pp = pprint.PrettyPrinter(indent=4)


class LidarRasterizer:
    def __init__(self, resolution, bbox, lidar, dimensions, filterexp):
        self.resolution = resolution
        self.bbox = bbox
        self.lidar = lidar
        self.dimensions = self._validate_dimensions(dimensions)
        self.reader = self._create_pipeline_reader(lidar)
        self.filterexp = filterexp
        self.logger = logging.getLogger(
            __name__
        )  # Is this the correct way to get logs inside instance of class ?

    def start(self):

        # Collect a writer for each dimension
        writers = [
            self._create_pipeline_writer(dim, self.resolution, self.bbox)
            for dim in self.dimensions
        ]

        rangefilter = {
            "type": "filters.range",
            "limits": "Classification[2:9]",  # Ground classification
        }

        smrf_filter = {
            "type": "filters.smrf",
            "ignore": "Classification[6:6]",
            "slope": 0.2,
            "window": 16,
            "threshold": 0.45,
            "scalar": 1.2,
        }
        # Build the pipeline by concating the reader, filter and writers
        pipeline_dict = {"pipeline": [self.reader, rangefilter] + writers}

        # Convert the pipeline to stringified JSON (required by PDAL)
        pipeline = pdal.Pipeline(json.dumps(pipeline_dict))

        if pipeline.validate():
            pipeline.loglevel = 8  # really noisy
            pipeline.execute()

        else:
            print("Pipeline not valid")

    @classmethod
    def _create_pipeline_reader(cls, lidar):
        reader = {"type": "readers.las", "filename": lidar}
        return reader

    @classmethod
    def _create_pipeline_writer(cls, dimension, resolution, bbox):
        # Convert the bbox from list [xmin,xmax,ymin,ymax] to: ([xmin,xmax],[ymin,ymax])
        bbox = str((bbox[0:2], bbox[2:4]))
        writer = {
            "type": "writers.gdal",
            "filename": dimension + ".tif",
            "resolution": resolution,
            "dimension": dimension,
            # "bounds": bbox,
            "gdaldriver": "GTiff",
            "gdalopts": "tiled=yes,compress=lzw,predictor=3",
            "output_type": "mean",
        }
        return writer

    @classmethod
    def _validate_dimensions(cls, dimensions):
        """Validates the dimensions given, against PDAL"""
        try:
            for dim in dimensions:
                if (
                    any(
                        pdaldim["name"] == dim
                        for pdaldim in pdal.dimension.getDimensions()
                    )
                    or dim == "Pulse width"
                ):
                    continue
                else:
                    raise ValueError(dim, "Dimension not recognized by PDAL")
            return dimensions
        except ValueError as e:
            print("ValueError: ", e)


def test():
    """ Only used for internal testing """
    resolution = 2  # Coarse resolution for fast testing
    bbox = [666000, 6666000, 667000, 6667000]
    lidar = os.path.join("testdata", "1km_6136_591.laz")
    dimensions = ["Z", "Intensity", "Amplitude", "Pulse width"]
    r = LidarRasterizer(resolution, bbox, lidar, dimensions, "")

    r.start()


if __name__ == "__main__":
    test()
