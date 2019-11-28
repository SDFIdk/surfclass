import numpy as np
from scipy import ndimage as nd


def fill_nearest_neighbor(a):
    """Fills the masked areas with the value of the nearest non-masked cell
    a: masked array
    returns: a non-masked ndarray
    """
    if not isinstance(a, np.ma.MaskedArray):
        raise TypeError("Input must be masked array")
    indexes = nd.distance_transform_edt(
        a.mask, return_indices=True, return_distances=False
    )
    filled = a.data[tuple(indexes)]
    return filled


def sieve(a, min_cluster_size, structure=np.ones((3, 3))):
    """Removes clusters smaller than a threshold size. Inplace
    a: masked array
    structure: array_like, optional. Defines feature connections. Default is 8-connectedness
    min_cluster_size: minimum number of cells to preserve a cluster.
    """
    if not isinstance(a, np.ma.MaskedArray):
        raise TypeError("Input must be masked array")
    class_values = np.unique(a.compressed())
    for c in class_values:
        mask = sieve_mask(a.data, c, min_cluster_size, structure=structure)
        a[mask] = np.ma.masked


def sieve_mask(a, class_number, min_cluster_size, structure=np.ones((3, 3))):
    """Gets a mask indicating clusters af given class smaller than a threshold size"""
    class_bin = a == class_number
    labeled_array, _ = nd.measurements.label(class_bin, structure)
    binc = np.bincount(labeled_array.ravel())
    noise_idx = np.where(binc < min_cluster_size)
    shp = a.shape
    mask = np.in1d(labeled_array, noise_idx).reshape(shp)
    return mask
