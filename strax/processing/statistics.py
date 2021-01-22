import numpy as np
import numba

import strax
export, __all__ = strax.exporter()

@export
@numba.njit()
def highest_density_region(data, fractions_desired, _buffer_size=10):
    """
    Computes for a given sampled distribution the highest density region
    of the desired fractions.

    Does not assume anything on the normalisation of the data.

    :param data: Sampled distribution
    :param fractions_desired: numpy.array Area/probability for which
        the hdr should be computed.
    :param _buffer_size: Size of the result buffer. The size is
        equivalent to the maximal number of allowed intervals.

    :return: two arrays: The first one stores the start and inclusive
        endindex of the highest density region. The second array holds
        the amplitude for which the desired fraction was reached.

    Note:
        Also goes by the name highest posterior density. Please note,
        that the right edge corresponds to the right side of the sample.
        Hence the corresponding index is -= 1.
    """
    fi = 0  # number of fractions seen
    # Buffer for the result if we find more then _buffer_size edges the function fails.
    # User can then manually increase the buffer if needed.
    res = np.zeros((len(fractions_desired), 2, _buffer_size), dtype=np.int32)
    res_amp = np.zeros(len(fractions_desired), dtype=np.float32)

    area_tot = np.sum(data)
    if area_tot <= 0:
        raise ValueError('Highest density regions are not defined for distributions '
                         'with a total probability of less-equal 0.')

    # Need an index which sorted by amplitude
    max_to_min = np.argsort(data)[::-1]

    lowest_sample_seen = np.inf
    for j in range(1, len(data)):
        # Loop over indices compute fractions from max to min
        if lowest_sample_seen == data[max_to_min[j]]:
            # We saw this sample height already, so no need to repeat
            continue

        lowest_sample_seen = data[max_to_min[j]]
        sorted_data_max_to_j = data[max_to_min[:j]]
        fraction_seen = np.sum(sorted_data_max_to_j - lowest_sample_seen) / area_tot

        # Check if this height step exceeded at least one of the desired
        # fractions
        m = fractions_desired[fi:] <= fraction_seen
        if not np.any(m):
            # If we do not exceed go to the next sample.
            continue

        for fraction_desired in fractions_desired[fi:fi + np.sum(m)]:
            # Since we loop always to the height of the next highest sample
            # it might happen that we overshoot the desired fraction. Similar
            # to the area deciles algorithm we have now to figure out at which
            # height we actually reached the desired fraction and store the
            # corresponding height:
            g = fraction_desired / fraction_seen

            # The following gives the true height, to get here one has to
            # solve for h:
            # 1. fraction_seen = sum_{i=0}^j (y_i - y_j) / a_total
            # 2. fraction_desired = sum_{i=0}^j (y_i - h) / a_total
            # 3. g = fraction_desired/fraction_seen
            # j == number of seen samples
            # n == number of total samples in distribution
            true_height = (1 - g) * np.sum(sorted_data_max_to_j) / j + g * lowest_sample_seen
            res_amp[fi] = true_height

            # Find gaps and get edges of hdr intervals:
            ind = np.sort(max_to_min[:j])
            gaps = np.arange(1, len(ind) + 1)

            g0 = 0
            g_ind = -1
            diff = ind[1:] - ind[:-1]
            gaps = gaps[:-1][diff > 1]
            if len(gaps):
                mes = ('Found more intervals than "_buffer_size" allows please'
                       'increase the size of the buffer.')
                assert len(gaps) < _buffer_size, mes

                for g_ind, g in enumerate(gaps):
                    # Loop over all gaps and get outer edges:
                    interval = ind[g0:g]
                    res[fi, 0, g_ind] = interval[0]
                    res[fi, 1, g_ind] = interval[-1] + 1
                    g0 = g

            # Now we have to do the last interval:
            interval = ind[g0:]
            res[fi, 0, g_ind + 1] = interval[0]
            res[fi, 1, g_ind + 1] = interval[-1] + 1

            fi += 1

        if fi == (len(fractions_desired)):
            # Found all fractions so we are done
            return res, res_amp
    
    # If we end up here this might be due to an offset 
    # of the distribution with respect to zero. In that case it can
    # happen that we do not find all desired fractions.
    # Hence we have to enforce to compute the last step from the last
    # lowest hight we have seen to zero.
    # Left and right edge is by definition 0 and len(data):
    res[fi:, 0, 0] = 0
    res[fi:, 1, 0] = len(data)
    # Now we have to compute the heights for the fractions we have not 
    # seen yet, since lowest_sample_seen == 0 and j == len(data)
    # the formula above reduces to:
    for ind, fraction_desired in enumerate(fractions_desired[fi:]):
        res_amp[fi+ind] = (1-fraction_desired) * np.sum(data)/len(data)
    return res, res_amp
