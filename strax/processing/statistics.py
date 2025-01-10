import numpy as np
import numba


import strax
from strax.sort_enforcement import stable_argsort, stable_sort

export, __all__ = strax.exporter()


@export
@numba.njit(nogil=True, cache=True)
def _compute_hdr_core(data, fractions_desired, only_upper_part=False, _buffer_size=10):
    """Core computation for highest density region initialization."""
    fi = 0
    res = np.zeros((len(fractions_desired), 2, _buffer_size), dtype=np.int32)
    res_amp = np.zeros(len(fractions_desired), dtype=np.float32)

    area_tot = np.sum(data)
    if area_tot <= 0:
        raise ValueError(
            "Highest density regions are not defined for distributions "
            "with a total probability of less-equal 0."
        )

    max_to_min = stable_argsort(data)[::-1]
    return max_to_min, area_tot, res, res_amp, fi


@export
@numba.njit(nogil=True, cache=True)
def _process_intervals_numba(ind, gaps, fi, res, g0, _buffer_size):
    """Process intervals using numba.

    Args:
        ind: Sorted indices
        gaps: Gap indices
        fi: Current fraction index
        res: Result buffer
        g0: Start index
        _buffer_size: Maximum number of intervals

    Returns:
        tuple: (fi + 1, res) Updated fraction index and result buffer

    """
    if len(gaps) > _buffer_size:
        res[fi, 0, :] = -1
        res[fi, 1, :] = -1
        return fi + 1, res

    g_ind = -1
    for g_ind, g in enumerate(gaps):
        interval = ind[g0:g]
        res[fi, 0, g_ind] = interval[0]
        res[fi, 1, g_ind] = interval[-1] + 1
        g0 = g

    interval = ind[g0:]
    res[fi, 0, g_ind + 1] = interval[0]
    res[fi, 1, g_ind + 1] = interval[-1] + 1
    return fi + 1, res


@export
@numba.njit(nogil=True, cache=True)
def _compute_fraction_seen(data, max_to_min, j, lowest_sample_seen, area_tot, only_upper_part):
    """Compute fraction seen (numba-compilable part).

    Args:
        data: Input distribution
        max_to_min: Sorted indices from max to min
        j: Current index
        lowest_sample_seen: Current lowest sample
        area_tot: Total area
        only_upper_part: If True, only compute area between max and current height

    Returns:
        tuple: (fraction_seen, sorted_data_max_to_j, actual_lowest)

    """
    lowest_sample_seen *= int(only_upper_part)
    sorted_data_max_to_j = data[max_to_min[:j]]
    return (
        np.sum(sorted_data_max_to_j - lowest_sample_seen) / area_tot,
        sorted_data_max_to_j,
        lowest_sample_seen,
    )


@export
@numba.njit(nogil=True, cache=True)
def _compute_true_height(sorted_data_sum, j, g, lowest_sample_seen):
    """Compute true height (numba-compilable part).

    Args:
        sorted_data_sum: Sum of sorted data
        j: Current index
        g: Fraction ratio
        lowest_sample_seen: Current lowest sample

    Returns:
        float: True height value

    """
    return (1 - g) * sorted_data_sum / j + g * lowest_sample_seen


@export
def highest_density_region(data, fractions_desired, only_upper_part=False, _buffer_size=10):
    """Compute highest density region for a given sampled distribution.

    This function splits only the stable sort operation into Python, keeping all other
    computations numba-accelerated for maximum performance.

    Args:
        data: Sampled distribution
        fractions_desired: Area/probability for which HDR should be computed
        only_upper_part: If True, only compute area between max and current height
        _buffer_size: Size of result buffer (max number of allowed intervals)

    Returns:
        tuple: (res, res_amp) where res contains interval indices and res_amp contains
               amplitudes for desired fractions

    """
    # Initialize using numba
    max_to_min, area_tot, res, res_amp, fi = _compute_hdr_core(
        data, fractions_desired, only_upper_part, _buffer_size
    )

    lowest_sample_seen = np.inf
    for j in range(1, len(data)):
        if lowest_sample_seen == data[max_to_min[j]]:
            continue

        lowest_sample_seen = data[max_to_min[j]]

        # Compute fraction seen (numba)
        fraction_seen, sorted_data_max_to_j, actual_lowest = _compute_fraction_seen(
            data, max_to_min, j, lowest_sample_seen, area_tot, only_upper_part
        )

        m = fractions_desired[fi:] <= fraction_seen
        if not np.any(m):
            continue

        for fraction_desired in fractions_desired[fi : fi + np.sum(m)]:
            g = fraction_desired / fraction_seen
            # Compute true height (numba)
            true_height = _compute_true_height(np.sum(sorted_data_max_to_j), j, g, actual_lowest)
            res_amp[fi] = true_height

            # Only stable_sort in Python mode
            with numba.objmode(ind="int64[:]"):
                ind = stable_sort(max_to_min[:j])

            # Rest stays in numba mode
            gaps = np.arange(1, len(ind) + 1)
            diff = ind[1:] - ind[:-1]
            gaps = gaps[:-1][diff > 1]

            # Process intervals with numba
            fi, res = _process_intervals_numba(ind, gaps, fi, res, 0, _buffer_size)

        if fi == len(fractions_desired):
            return res, res_amp

    # Handle remaining fractions (in numba)
    res[fi:, 0, 0] = 0
    res[fi:, 1, 0] = len(data)
    for ind, fraction_desired in enumerate(fractions_desired[fi:]):
        res_amp[fi + ind] = (1 - fraction_desired) * np.sum(data) / len(data)

    return res, res_amp
