import numpy as np
from scipy import ndimage as nd


def fill_nearest_neighbor(a):
    """Fills the masked areas of a masked array with the value of the nearest non-masked cell"""
    if not isinstance(a, np.ma.MaskedArray):
        raise TypeError("Input must be masked array")
    indexes = nd.distance_transform_edt(
        a.mask, return_indices=True, return_distances=False
    )
    filled = a.data[tuple(indexes)]
    return filled
