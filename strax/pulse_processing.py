"""Functions that perform basic pulse processing
"""
import numpy as np
import numba

from . import utils
from .data import hit_dtype

__all__ = 'sort_by_time baseline coincidence_level find_hits'.split()

# ~7x faster than np.sort(records, order='time'). Try it.
@numba.jit(nopython=True)
def sort_by_time(records):
    time = records['time'].copy()
    sort_i = np.argsort(time)
    return records[sort_i]


@numba.jit(nopython=True)
def baseline(records, baseline_samples=40):
    """Subtract pulses from int(baseline), store baseline in baseline field
    :param baseline_samples: number of samples at start of pulse to average
    Assumes records are sorted in time (or at least by channel, then time)
    """
    if not len(records):
        return
    samples_per_record = len(records[0]['data'])

    # Array for looking up last baseline seen in channel
    # We only care about the channels in this set of records; a single .max()
    # is worth avoiding the hassle of passing n_channels around
    last_bl_in = np.zeros(records['channel'].max() + 1, dtype=np.int16)

    for d_i, d in enumerate(records):

        # Compute the baseline if we're the first record of the pulse,
        # otherwise take the last baseline we've seen in the channel
        if d.record_i == 0:
            bl = last_bl_in[d.channel] = d.data[:baseline_samples].mean()
        else:
            bl = last_bl_in[d.channel]

        # Subtract baseline from all data samples in the record
        # (any additional zeros are already zero)
        last = min(samples_per_record,
                   d.total_length - d.record_i * samples_per_record)
        d.data[:last] = int(bl) - d.data[:last]
        d.baseline = bl


@numba.jit(nopython=True)
def coincidence_level(records, dt):
    """Return number of records that start dt samples earlier or later
     (including itself, i.e. the minimum value is 1)
     TODO: inclusive or exclusive bounds?

     records MUST be sorted by time!
     """
    i_left = 0
    i_right = 0
    i_center = 0
    t = records['time']
    n_max = len(records) - 1
    result = np.zeros(len(records))

    while True:
        if t[i_right] - dt < t[i_center] and i_right < n_max:
            i_right += 1
            continue
        # If here, right edge is beyond coincidence limit
        # (or cannot extend)

        if t[i_left] + dt < t[i_center] and i_left < n_max:
            i_left += 1
            continue
        # If here, left edge is inside coincidence limit
        # (or cannot extend - should never happen -- assert?)

        # Note no +1: right edge is one too far
        result[i_center] = i_right - i_left
        if i_center < n_max:
            i_center += 1
        else:
            break

    return result


# Chunk size should be at least > 1000,
# else copying buffers / switching context dominates over actual computation
@utils.growing_result(hit_dtype, chunk_size=int(1e4))
@numba.jit(nopython=True)
def find_hits(result_buffer, records, threshold=15):
    if not len(records):
        return
    samples_per_record = len(records[0]['data'])
    offset = 0

    for r in records:
        in_interval = False
        hit_start = -1

        for i in range(len(r['data'])):
            # We can't use enumerate over r['data'], numba gives error
            # TODO: file issue?
            above_threshold = r['data'][i] > threshold

            if not in_interval and above_threshold:
                # Start of a hit
                in_interval = True
                hit_start = i

            if in_interval and (not above_threshold
                                or i == samples_per_record):
                # End of the current hit
                in_interval = False

                # The hit ended just before this index
                # ... except if this is the last sample in the record and
                # we're still above threshold. Then the hit ends right here.
                hit_end = i - 1 if not above_threshold else i

                # Add bounds to result buffer
                result_buffer[offset]['left'] = r['time'] + hit_start
                result_buffer[offset]['right'] = r['time'] + hit_end
                result_buffer[offset]['channel'] = r['channel']

                offset += 1
                if offset == len(result_buffer):
                    yield offset
                    offset = 0
    yield offset


find_hits.__doc__ = """
Return hits (intervals above threshold) found in records.
Hits that straddle record boundaries are split (TODO: fix this?)
"""
