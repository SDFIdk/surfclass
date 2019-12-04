"""Tools for handling noisy output from classification."""
import numpy as np
from scipy import ndimage as nd
from skimage.filters import rank


def fill_nearest_neighbor(a):
    """Fills masked cells with value from nearest non-masked cell.

    Args:
        a (MaskedArray): A 2D array.

    Raises:
        TypeError: If `a` is not a MaskedArray

    Returns:
        ndarray: A 2D array.

    """
    if not isinstance(a, np.ma.MaskedArray):
        raise TypeError("Input must be masked array")
    if not np.ma.is_masked(a):
        return a
    indexes = nd.distance_transform_edt(
        a.mask, return_indices=True, return_distances=False
    )
    filled = a.data[tuple(indexes)]
    return filled


def sieve(a, min_cluster_size, structure=np.ones((3, 3))):
    """Masks clusters smaller than a threshold size.

    A cluster is a group of cells connected to each other as defined by `structure`.

    Note:
        Changes input array.

    Args:
        a (MaskedArray): 2D array.
        min_cluster_size (int): Minimum size (in number of cells) to keep a cluster. Clusters smaller than
            this threshold will be masked.
        structure (ndarray, optional): The neighborhood expressed as a 2-D array of 1’s and 0’s. Defaults to
            np.ones((3, 3)) which is 8-connectedness.

    Raises:
        TypeError: If input is not a MaskedArray.

    """
    if not isinstance(a, np.ma.MaskedArray):
        raise TypeError("Input must be masked array")
    class_values = np.unique(a.compressed())
    for c in class_values:
        mask = sieve_mask(a.data, c, min_cluster_size, structure=structure)
        a[mask] = np.ma.masked


def sieve_mask(a, class_number, min_cluster_size, structure=np.ones((3, 3))):
    """Gets a bool mask indicating clusters of given cell value smaller than a threshold size.

    Args:
        a (ndarray): 2D array.
        class_number (number): Cell value.
        min_cluster_size (int): Minimum size (in number of cells) to keep a cluster. Clusters smaller than
            this threshold will be masked.
        structure (ndarray, optional): The neighborhood expressed as a 2-D array of 1’s and 0’s. Defaults to
            np.ones((3, 3)) which is 8-connectedness.

    Returns:
        [ndarray]: 2D array of bools with the same shape as input array.

    """
    class_bin = a == class_number
    labeled_array, _ = nd.measurements.label(class_bin, structure)
    binc = np.bincount(labeled_array.ravel())
    noise_idx = np.where(binc < min_cluster_size)
    shp = a.shape
    mask = np.in1d(labeled_array, noise_idx).reshape(shp)
    return mask


def majority_vote(a, iterations=1, structure=np.ones((3, 3))):
    """Changes cell values to the most frequent value in its neighborhood.

    Args:
        a (ndarray): 2D ndarray. Possible a MaskedArray.
        iterations (int, optional): Number of times to repeat the process. Defaults to 1.
        structure (ndarray, optional): The neighborhood expressed as a 2-D array of 1’s and 0’s. Defaults to
            np.ones((3, 3)) which is 8-connectedness.

    Returns:
        ndarray: 2D ndarray of same dimensions as input array. MaskedArray if input is masked.

    """
    nodata = None
    assert a.dtype == "uint8", "Majority vote only works for uint8"
    if np.ma.is_masked(a):
        # windowed_histogram does not work with masked arrays
        nodata = np.max(a) + 1
        a = a.filled(nodata)
    for _ in range(iterations):
        a = rank.windowed_histogram(a, structure).argmax(axis=-1).astype("uint8")
    return np.ma.masked_values(a, nodata) if nodata is not None else a


def denoise(a):
    """Applies simple denoising to a classified raster.

    Denoising removes small clusters and fills nodata areas.

    Args:
        a (MaskedArray): 2D MaskedArray with 'uint8' type

    Returns:
        ndarray: Denoised data

    """
    a = majority_vote(a, 2)
    a = fill_nearest_neighbor(a)
    denoised = majority_vote(a, 1)
    return denoised.filled() if isinstance(denoised, np.ma.MaskedArray) else denoised
