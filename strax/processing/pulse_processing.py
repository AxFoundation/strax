"""Functions that perform processing on pulses
(other than data reduction functions, which are in data_reduction.py)
"""
import numpy as np
import numba

import strax
export, __all__ = strax.exporter()

# Constant for use in record_links, to indicate there is no prev/next record
NOT_APPLICABLE = -1


@export
@numba.jit(nopython=True, nogil=True, cache=True)
def baseline(records, baseline_samples=40):
    """Subtract pulses from int(baseline), store baseline in baseline field
    :param baseline_samples: number of samples at start of pulse to average
    Assumes records are sorted in time (or at least by channel, then time)

    Assumes record_i information is accurate (so don't cut pulses before
    baselining them!)
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
        # (any additional zeros should be kept at zero)
        last = min(samples_per_record,
                   d.pulse_length - d.record_i * samples_per_record)
        d.data[:last] = int(bl) - d.data[:last]
        d.baseline = bl


@export
@numba.jit(nopython=True, nogil=True, cache=True)
def integrate(records):
    for i, r in enumerate(records):
        records[i]['area'] = r['data'].sum()


@export
@numba.jit(nopython=True, nogil=True, cache=True)
def record_links(records):
    """Return (prev_r, next_r), each arrays of indices of previous/next
    record in the same pulse, or -1 if this is not applicable
    """
    # TODO: needs tests
    if not len(records):
        return
    n_channels = records['channel'].max() + 1
    samples_per_record = len(records[0]['data'])
    previous_record = np.ones(len(records), dtype=np.int32) * NOT_APPLICABLE
    next_record = np.ones(len(records), dtype=np.int32) * NOT_APPLICABLE

    # What was the index of the last record seen in each channel?
    last_record_seen = np.ones(n_channels, dtype=np.int32) * NOT_APPLICABLE
    # What would the start time be of a record that continues that record?
    expected_next_start = np.zeros(n_channels, dtype=np.int64)

    for i, r in enumerate(records):
        ch = r['channel']
        last_i = last_record_seen[ch]
        if r['record_i'] == 0:
            # Record starts a new pulse
            previous_record[i] = NOT_APPLICABLE

        elif r['time'] == expected_next_start[ch]:
            # Continuing record.
            previous_record[i] = last_i
            next_record[last_i] = i

        # (If neither matches, this is a continuing record, but the starting
        #  record has been cut away (e.g. for data reduction))

        last_record_seen[ch] = i
        expected_next_start[ch] = samples_per_record * r['dt']

    return previous_record, next_record


# Chunk size should be at least a thousand,
# else copying buffers / switching context dominates over actual computation
# No max_duration argument: hits terminate at record boundaries, and
# anyone insane enough to try O(sec) long records deserves to be punished
@export
@strax.growing_result(strax.hit_dtype, chunk_size=int(1e4))
@numba.jit(nopython=True, nogil=True, cache=True)
def find_hits(records, threshold=15, _result_buffer=None):
    """Return hits (intervals above threshold) found in records.
    Hits that straddle record boundaries are split (TODO: fix this?)

    NB: returned hits are NOT sorted yet!
    """
    buffer = _result_buffer
    if not len(records):
        return
    samples_per_record = len(records[0]['data'])
    offset = 0

    for record_i, r in enumerate(records):
        # print("Starting record ', record_i)
        in_interval = False
        hit_start = -1

        for i in range(samples_per_record):
            # We can't use enumerate over r['data'], numba gives error
            # TODO: file issue?
            above_threshold = r['data'][i] > threshold
            # print(r['data'][i], above_threshold, in_interval, hit_start)

            if not in_interval and above_threshold:
                # Start of a hit
                in_interval = True
                hit_start = i

            if in_interval:
                if not above_threshold:
                    # Hit ends at the start of this sample
                    hit_end = i
                    in_interval = False

                elif i == samples_per_record - 1:
                    # Hit ends at the *end* of this sample
                    # (because the record ends)
                    hit_end = i + 1
                    in_interval = False

                if not in_interval:
                    # print('saving hit')
                    # Hit is done, add it to the result
                    res = buffer[offset]
                    res['left'] = hit_start
                    res['right'] = hit_end
                    res['time'] = r['time'] + hit_start * r['dt']
                    # Note right bound is exclusive, no + 1 here:
                    res['length'] = hit_end - hit_start
                    res['dt'] = r['dt']
                    res['channel'] = r['channel']
                    res['record_i'] = record_i

                    # Yield buffer to caller if needed
                    offset += 1
                    if offset == len(buffer):
                        yield offset
                        offset = 0

                    # Clear stuff, just for easier debugging
                    # hit_start = 0
                    # hit_end = 0
    yield offset
