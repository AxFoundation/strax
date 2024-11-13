import numpy as np
import numba
from numba.extending import register_jitable

import strax
from strax.sort_enforcement import stable_argsort, stable_sort

export, __all__ = strax.exporter()


@export
@register_jitable
def _compute_hdr_core(data, fractions_desired, only_upper_part=False, _buffer_size=10):
    """Core computation for highest density region.
    Returns the data needed for interval computation and the result arrays.
    """
    fi = 0  # number of fractions seen
    res = np.zeros((len(fractions_desired), 2, _buffer_size), dtype=np.int32)
    res_amp = np.zeros(len(fractions_desired), dtype=np.float32)

    area_tot = np.sum(data)
    if area_tot <= 0:
        raise ValueError(
            "Highest density regions are not defined for distributions "
            "with a total probability of less-equal 0."
        )

    # Need an index which sorted by amplitude
    max_to_min = stable_argsort(data)[::-1]
    return max_to_min, area_tot, res, res_amp, fi

@export
def _process_hdr_intervals(ind, gaps, fi, res, g0, _buffer_size):
    """Process the intervals for highest density region.
    This function handles the stable sorting part outside of numba.
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
    
    # Last interval
    interval = ind[g0:]
    res[fi, 0, g_ind + 1] = interval[0]
    res[fi, 1, g_ind + 1] = interval[-1] + 1
    return fi + 1, res

@export
@register_jitable
def highest_density_region(data, fractions_desired, only_upper_part=False, _buffer_size=10):
    """Computes for a given sampled distribution the highest density region of the desired
    fractions. Does not assume anything on the normalisation of the data.
    
    :param data: Sampled distribution
    :param fractions_desired: numpy.array Area/probability for which
        the hdr should be computed.
    :param _buffer_size: Size of the result buffer. The size is
        equivalent to the maximal number of allowed intervals.
    :param only_upper_part: Boolean, if true only computes
        area/probability between maximum and current height.
    :return: two arrays: The first one stores the start and inclusive
        endindex of the highest density region. The second array holds
        the amplitude for which the desired fraction was reached.
    """
    max_to_min, area_tot, res, res_amp, fi = _compute_hdr_core(
        data, fractions_desired, only_upper_part, _buffer_size)
    
    lowest_sample_seen = np.inf
    for j in range(1, len(data)):
        if lowest_sample_seen == data[max_to_min[j]]:
            continue

        lowest_sample_seen = data[max_to_min[j]]
        lowest_sample_seen *= int(only_upper_part)
        sorted_data_max_to_j = data[max_to_min[:j]]
        fraction_seen = np.sum(sorted_data_max_to_j - lowest_sample_seen) / area_tot

        m = fractions_desired[fi:] <= fraction_seen
        if not np.any(m):
            continue

        for fraction_desired in fractions_desired[fi : fi + np.sum(m)]:
            g = fraction_desired / fraction_seen
            true_height = (1 - g) * np.sum(sorted_data_max_to_j) / j + g * lowest_sample_seen
            res_amp[fi] = true_height

            # This part needs stable_sort - switch to object mode
            with numba.objmode(ind='int64[:]'):
                ind = stable_sort(max_to_min[:j])
            
            gaps = np.arange(1, len(ind) + 1)
            diff = ind[1:] - ind[:-1]
            gaps = gaps[:-1][diff > 1]

            # Process intervals outside numba
            with numba.objmode(fi='int64', res='int32[:, :, :]'):
                fi, res = _process_hdr_intervals(ind, gaps, fi, res, 0, _buffer_size)

            if fi == len(fractions_desired):
                return res, res_amp

    # Handle remaining fractions
    with numba.objmode():
        res[fi:, 0, 0] = 0
        res[fi:, 1, 0] = len(data)
        for ind, fraction_desired in enumerate(fractions_desired[fi:]):
            res_amp[fi + ind] = (1 - fraction_desired) * np.sum(data) / len(data)
    
    return res, res_amp