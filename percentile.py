import numpy as np
import os

def approximate_percentile_from_npy_files(
    file_list,
    q,
    bins=65536,
    value_range=None,
):
    """
    Approximate percentile over many .npy image files using a histogram.

    Args:
        file_list: list of .npy file paths, each containing one image
        q: percentile in [0, 100]
        bins: histogram bins
        value_range: (min_val, max_val). If None, do a first pass to estimate exactly.

    Returns:
        approx percentile value
    """
    if value_range is None:
        global_min = np.inf
        global_max = -np.inf
        for f in file_list:
            x = np.load(f, mmap_mode="r")
            global_min = min(global_min, float(np.min(x)))
            global_max = max(global_max, float(np.max(x)))
        value_range = (global_min, global_max)

    hist = np.zeros(bins, dtype=np.int64)

    for f in file_list:
        x = np.load(f, mmap_mode="r")
        h, edges = np.histogram(x, bins=bins, range=value_range)
        hist += h

    cdf = np.cumsum(hist)
    total = cdf[-1]
    target = q / 100.0 * total
    idx = np.searchsorted(cdf, target)

    bin_edges = edges
    idx = min(idx, len(bin_edges) - 2)
    left = bin_edges[idx]
    right = bin_edges[idx + 1]

    return 0.5 * (left + right)