"""Functions that perform processing on pulses
(other than data reduction functions, which are in data_reduction.py)
"""
import numpy as np
import numba
from scipy.ndimage import convolve1d

import strax
export, __all__ = strax.exporter()
__all__ += ['NO_RECORD_LINK']

# Constant for use in record_links, to indicate there is no prev/next record
NO_RECORD_LINK = -1


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
        return records
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
def zero_out_of_bounds(records):
    """"Set waveforms to zero out of pulse bounds
    """
    if not len(records):
        return records
    samples_per_record = len(records[0]['data'])

    for r in records:
        end = r['pulse_length'] - r['record_i'] * samples_per_record
        if end < samples_per_record:
            r['data'][end:] = 0


@export
@numba.jit(nopython=True, nogil=True, cache=True)
def integrate(records):
    if not len(records):
        return
    samples_per_record = len(records[0]['data'])
    for i, r in enumerate(records):
        n_real_samples = min(
            samples_per_record,
            r['pulse_length'] - r['record_i'] * samples_per_record)
        records[i]['area'] = (
            r['data'].sum()
            + int(round(r['baseline'] % 1)) * n_real_samples)



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
    previous_record = np.ones(len(records), dtype=np.int32) * NO_RECORD_LINK
    next_record = np.ones(len(records), dtype=np.int32) * NO_RECORD_LINK

    # What was the index of the last record seen in each channel?
    last_record_seen = np.ones(n_channels, dtype=np.int32) * NO_RECORD_LINK
    # What would the start time be of a record that continues that record?
    expected_next_start = np.zeros(n_channels, dtype=np.int64)

    for i, r in enumerate(records):
        ch = r['channel']
        if ch < 0:
            # We can't print the channel number in the exception message
            # from numba, hence the extra print here.
            print("Found negative channel number")
            print(ch)
            raise ValueError("Negative channel number?!")
        last_i = last_record_seen[ch]

        # if r['time'] < expected_next_start[ch]:
        #     print(r['time'], expected_next_start[ch], ch)
        #     raise ValueError("Overlapping pulse found!")

        if r['record_i'] == 0:
            # Record starts a new pulse
            previous_record[i] = NO_RECORD_LINK

        elif r['time'] == expected_next_start[ch]:
            # Continuing record.
            previous_record[i] = last_i
            next_record[last_i] = i

        # (If neither matches, this is a continuing record, but the starting
        #  record has been cut away (e.g. for data reduction))
        last_record_seen[ch] = i
        expected_next_start[ch] = r['time'] + samples_per_record * r['dt']

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
        area = 0

        for i in range(samples_per_record):
            # We can't use enumerate over r['data'],
            # numba gives errors if we do.
            # TODO: file issue?
            x = r['data'][i]
            above_threshold = x > threshold
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
                    area += x
                    in_interval = False

                else:
                    area += x

                if not in_interval:
                    # print('saving hit')
                    # Hit is done, add it to the result
                    if hit_end == hit_start:
                        print(r['time'], r['channel'], hit_start)
                        raise ValueError(
                            "Caught attempt to save zero-length hit!")
                    res = buffer[offset]
                    res['left'] = hit_start
                    res['right'] = hit_end
                    res['time'] = r['time'] + hit_start * r['dt']
                    # Note right bound is exclusive, no + 1 here:
                    res['length'] = hit_end - hit_start
                    res['dt'] = r['dt']
                    res['channel'] = r['channel']
                    res['record_i'] = record_i
                    area += int(round(
                        res['length'] * (r['baseline'] % 1)))
                    res['area'] = area
                    area = 0

                    # Yield buffer to caller if needed
                    offset += 1
                    if offset == len(buffer):
                        yield offset
                        offset = 0

                    # Clear stuff, just for easier debugging
                    # hit_start = 0
                    # hit_end = 0
    yield offset


@export
def filter_records(r, ir):
    """Apply filter with impulse response ir over the records r.
    Assumes the filter origin is at the impulse response maximum.

    :param ws: Waveform matrix, must be float
    :param ir: Impulse response, must have odd length. Will normalize.
    :param prev_r: Previous record map from strax.record_links
    :param next_r: Next record map from strax.record_links
    """
    # Convert waveforms to float and restore baseline
    ws = r['data'].astype(np.float) + (r['baseline'] % 1)[:, np.newaxis]

    prev_r, next_r = strax.record_links(r)
    ws_filtered = filter_waveforms(
        ws,
        ir / ir.sum(),
        prev_r, next_r)

    # Restore waveforms as integers
    r['data'] = ws_filtered.astype(np.int16)


@export
def filter_waveforms(ws, ir, prev_r, next_r):
    """Convolve filter with impulse response ir over each row of ws.
    Assumes the filter origin is at the impulse response maximum.

    :param ws: Waveform matrix, must be float
    :param ir: Impulse response, must have odd length.
    :param prev_r: Previous record map from strax.record_links
    :param next_r: Next record map from strax.record_links
    """
    n = len(ir)
    a = n//2
    if n % 2 == 0:
        raise ValueError("Impulse response must have odd length")

    # Do the convolutions outside numba;
    # numba supports np.convolve, but this seems to be quite slow

    # Main convolution
    maxi = np.argmax(ir)
    result = convolve1d(ws,
                        ir,
                        origin=maxi - a,
                        mode='constant')

    # Contribution to next record (if present)
    have_next = ws[next_r != -1]
    to_next = convolve1d(have_next[:, -(n - maxi - 1):],
                         ir,
                         origin=a,
                         mode='constant')

    # Contribution to previous record (if present)
    have_prev = ws[prev_r != -1]
    to_prev = convolve1d(have_prev[:, :maxi],
                         ir,
                         origin=-a,
                         mode='constant')

    # Combine the results in numba; here numba is much faster (~100x?)
    # than a numpy assignment using boolean array instead of a for loop.
    _combine_filter_results(result, to_next, to_prev, next_r, prev_r, maxi, n)
    return result


@numba.jit(nopython=True, cache=True, nogil=True)
def _combine_filter_results(result, to_next, to_prev, next_r, prev_r, maxi, n):
    seen_that_have_next = 0
    seen_that_have_prev = 0
    for i in range(len(result)):
        if next_r[i] != NO_RECORD_LINK:
            result[next_r[i], :n - maxi - 1] += to_next[seen_that_have_next]
            seen_that_have_next += 1
        if prev_r[i] != NO_RECORD_LINK:
            result[prev_r[i], -maxi:] += to_prev[seen_that_have_prev]
            seen_that_have_prev += 1
