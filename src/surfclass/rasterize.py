import os
import pdal


class Rasterizer:
    def __init__(self, resolution, bbox, lidar, dimensions, filterexp):
        self.resolution = (resolution,)
        self.bbox = (bbox,)
        self.lidar = (lidar,)
        self.dimensions = self._validate_dimensions(dimensions)
        # self.reader = self._create_pipeline_reader(self)
        self.filterexp = filterexp

    def start(self):
        print("Starting rasterizer")

    @classmethod
    def _validate_dimensions(cls, dimensions):
        """Validates the dimensions given, against PDAL"""
        try:
            for dim in dimensions:
                if any(
                    pdaldim["name"] == dim for pdaldim in pdal.dimension.getDimensions()
                ):
                    continue
                else:
                    raise ValueError(dim, " Dimension not recognized by PDAL")
            return dimensions
        except ValueError as e:
            print("ValueError: ", e)


def test():
    """ Only used for internal testing """
    resolution = 0.4
    bbox = [666000, 6666000, 667000, 6667000]
    lidar = os.path.join("testdata", "1km_6136_592.laz ")
    dimensions = ["Z", "Amplitude"]
    r = Rasterizer(resolution, bbox, lidar, dimensions, "")
    print(r.dimensions)


if __name__ == "__main__":
    test()
