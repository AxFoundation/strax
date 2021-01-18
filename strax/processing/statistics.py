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
    :param fractions_desired: Area/probability for which the hdr should
        be computed.
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

    # Need index sorted by amplitude
    max_to_min = np.argsort(data)[::-1]

    lowest_point = np.inf
    for j in range(1, len(data) - 1):
        # Computing fractions from max to min
        if lowest_point == data[max_to_min[j]]:
            # We saw this sample height already, so no need to repeat
            continue

        lowest_point = data[max_to_min[j]]
        d = data[max_to_min[:j]]
        f = np.sum(d - lowest_point) / area_tot

        # Check if this height step exceeded at least one of the desired
        # fractions
        m = fractions_desired[fi:] <= f
        if np.any(m):
            for fd in fractions_desired[fi:fi + np.sum(m)]:
                # Compute correct height for which we reach desired
                # fraction and store:
                g = fd / f
                h = (1 - g) * np.sum(d) / j + g * lowest_point
                res_amp[fi] = h

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
            return res, res_amp
    raise ValueError('Have not found all the desired fractions.'
                     ' This should not have happened.')