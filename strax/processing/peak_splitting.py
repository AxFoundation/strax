"""Peak splitting functions

TODO: there is considerable duplication between the two algorithms
implemented here. Be careful when editing: if you fix a bug, there is
likely a mirrored but in the other algorithm.

(if you fix the duplication, you will be rewarded generously in the afterlife.)
"""


import numpy as np
import numba
import strax

export, __all__ = strax.exporter()


@export
def split_peaks(peaks, records, to_pe, algorithm='local_minimum',
                min_height=25, min_ratio=4,
                threshold=0.4, min_area=40,
                do_iterations=1):
    """Return peaks after splitting .
    :param peaks: peaks to split
    :param records: records from which peaks were built; for building waveforms of
    split peaks.
    :param algorithm: 'local_minimum' (default, for backwards compatibility)
    or 'natural_breaks' (should be better!)
    :param do_iterations: Maximum number of iterations / recursive splits to do.

    Other options depend on the algorithm.
        local_minimum: takes options min_height and min_ratio.
            Splits peaks at prominent local minima.
            On either side of a split point, local maxima are:
                - larger than minimum + min_height, AND
                - larger than minimum * min_ratio
            (this is related to topographical prominence for mountains)
            NB: Min_height is in pe/ns, NOT pe/bin!

        natural_breaks: takes options threshold and min_area
            Split peaks according to Jenks natural breaks algorithm,
            until all fragments have GOF < threshold, or < area.
    """
    if not len(records) or not len(peaks):
        return peaks

    while do_iterations > 0:
        is_split = np.zeros(len(peaks), dtype=np.bool_)

        # Support for both algorithms is a bit awkward since we rely on numba,
        # so we can't use many of python's cool tricks to avoid duplication
        # (there surely is a way, but I'm too lazy to find one right now)
        if algorithm == 'natural_breaks':
            new_peaks = _split_peaks_nb(
                peaks,
                orig_dt=records[0]['dt'],
                is_split=is_split,
                threshold=float(threshold),
                min_area=float(min_area),
                result_dtype=peaks.dtype)
        else:
            new_peaks = _split_peaks(
                peaks,
                orig_dt=records[0]['dt'],
                is_split=is_split,
                min_height=float(min_height),
                min_ratio=float(min_ratio),
                result_dtype=peaks.dtype)

        if is_split.sum() == 0:
            # Nothing was split -- end early
            break

        strax.sum_waveform(new_peaks, records, to_pe)
        peaks = strax.sort_by_time(np.concatenate([peaks[~is_split], new_peaks]))
        do_iterations -= 1

    return peaks



##
# Local minimum clustering
##


@strax.utils.growing_result(dtype=strax.peak_dtype(), chunk_size=int(1e4))
@numba.jit(nopython=True, nogil=True, cache=True)
def _split_peaks(peaks, orig_dt, is_split,
                 min_height, min_ratio,
                 _result_buffer=None, result_dtype=None):
    # TODO NEEDS TESTS!
    new_peaks = _result_buffer
    offset = 0

    for p_i, p in enumerate(peaks):
        prev_split_i = 0

        w = p['data'][:p['length']]

        for split_i in find_split_points(
                w,
                min_height=min_height * p['dt'],
                min_ratio=min_ratio):
            is_split[p_i] = True
            r = new_peaks[offset]
            r['time'] = p['time'] + prev_split_i * p['dt']
            r['channel'] = p['channel']
            # Set the dt to the original (lowest) dt first;
            # this may change when the sum waveform of the new peak is computed
            r['dt'] = orig_dt
            r['length'] = (split_i - prev_split_i) * p['dt'] / orig_dt

            if r['length'] <= 0:
                print(p['data'])
                print(prev_split_i, split_i)
                raise ValueError("Attempt to create invalid peak!")

            offset += 1
            if offset == len(new_peaks):
                yield offset
                offset = 0

            prev_split_i = split_i

    yield offset


@numba.jit(nopython=True, nogil=True, cache=True)
def find_split_points(w, min_height=0, min_ratio=0):
    """"Yield indices of prominent local minima in w
    If there was at least one index, yields len(w)-1 at the end
    """
    found_one = False
    last_max = -99999999999999.9
    min_since_max = 99999999999999.9
    min_since_max_i = 0

    for i, x in enumerate(w):
        if x < min_since_max:
            # New minimum since last max
            min_since_max = x
            min_since_max_i = i

        if min(last_max, x) > max(min_since_max + min_height,
                                  min_since_max * min_ratio):
            # Significant local minimum: tell caller,
            # reset both max and min finder
            yield min_since_max_i
            found_one = True
            last_max = x
            min_since_max = 99999999999999.9
            min_since_max_i = i

        if x > last_max:
            # New max, reset minimum finder state
            # Notice this is AFTER the split check,
            # to accomodate very fast rising second peaks
            last_max = x
            min_since_max = 99999999999999.9
            min_since_max_i = i

    if found_one:
        yield len(w)


##
# Natural breaks clustering
##

@strax.utils.growing_result(dtype=strax.peak_dtype(), chunk_size=int(1e4))
@numba.jit(nopython=True, nogil=True, cache=True)
def _split_peaks_nb(peaks, orig_dt, is_split,
                    # For natural breaks
                    threshold, min_area,
                    _result_buffer=None, result_dtype=None):
    # TODO NEEDS TESTS!
    # TODO any way to avoid the code duplication that is compatible with numba?
    new_peaks = _result_buffer
    offset = 0

    for p_i, p in enumerate(peaks):
        if p['area'] < min_area:
            continue

        prev_split_i = 0
        w = p['data'][:p['length']]

        for split_i in find_split_points_nb(
                w,
                threshold=threshold):
            is_split[p_i] = True
            r = new_peaks[offset]
            r['time'] = p['time'] + prev_split_i * p['dt']
            r['channel'] = p['channel']
            # Set the dt to the original (lowest) dt first;
            # this may change when the sum waveform of the new peak is computed
            r['dt'] = orig_dt
            r['length'] = (split_i - prev_split_i) * p['dt'] / orig_dt

            if r['length'] <= 0:
                print(p['data'])
                print(prev_split_i, split_i)
                raise ValueError("Attempt to create invalid peak!")

            offset += 1
            if offset == len(new_peaks):
                yield offset
                offset = 0

            prev_split_i = split_i

    yield offset


@numba.njit(nogil=True, cache=True)
def find_split_points_nb(w, threshold=0.4, _dummy=0.):
    gofs = natural_breaks_gof(w)
    i = np.argmax(gofs)
    if gofs[i] > threshold:
        yield i
        yield len(w) - 1


@export
@numba.njit(nogil=True, cache=True)
def natural_breaks_gof(w):
    """Return natural breaks goodness of split/fit for the waveform w
    a sharp peak gives ~0, two widely separate peaks ~1.
    """
    left = weighted_var_online(w)
    right = weighted_var_online(w[::-1])[::-1]
    gof = 1 - (left + right) / left[-1]
    return gof


@numba.njit(nogil=True, cache=True)
def weighted_var_online(waveform):
    """Return left-to-right result of an online weighted variance computation
    on the waveform.
    """
    mean = sum_weights = s = 0
    result = np.zeros(len(waveform))
    for i, w in enumerate(waveform):
        sum_weights += w
        if sum_weights == 0:
            continue

        mean_old = mean
        mean = mean_old + (w / sum_weights) * (i - mean_old)
        s += w * (i - mean_old) * (i - mean)
        result[i] = s / sum_weights

    return result
