import numpy as np
import numba
import strax

__all__ = 'split_peaks '.split()


def split_peaks(peaks, records, to_pe, min_height=25, min_ratio=4):
    """Return peaks after splitting at prominent sum waveform minima
    'Prominent' means: on either side of a split point, local maxima are:
    - larger than minimum + min_height
    - larger than minimum * min_ratio
    (this is related to topographical prominence for mountains)

    Min_height is in pe/ns (NOT pe/bin!)
    """
    if not len(records) or not len(peaks):
        # Empty chunk: cannot proceed
        return peaks
    
    is_split = np.zeros(len(peaks), dtype=np.bool_)

    new_peaks = _split_peaks(peaks,
                             min_height=min_height,
                             min_ratio=min_ratio,
                             orig_dt=records[0]['dt'],
                             is_split=is_split,
                             result_dtype=peaks.dtype)
    strax.sum_waveform(new_peaks, records, to_pe)
    return strax.sort_by_time(np.concatenate([peaks[~is_split],
                                              new_peaks]))


@strax.utils.growing_result(dtype=strax.peak_dtype(), chunk_size=int(1e4))
@numba.jit(nopython=True, nogil=True, cache=True)
def _split_peaks(peaks, min_height, min_ratio, orig_dt, is_split,
                 _result_buffer=None, result_dtype=None):
    # TODO NEEDS TESTS!
    new_peaks = _result_buffer
    offset = 0

    for p_i, p in enumerate(peaks):
        prev_split_i = 0

        for split_i in find_split_points(p['data'][:p['length']],
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
