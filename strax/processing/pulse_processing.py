"""Functions that perform processing on pulses
(other than data reduction functions, which are in data_reduction.py)
"""
import typing as ty

import numpy as np
import numba
from scipy.ndimage import convolve1d
from warnings import warn
import strax
export, __all__ = strax.exporter()
__all__ += ['NO_RECORD_LINK']

# Constant for use in record_links, to indicate there is no prev/next record
NO_RECORD_LINK = -1


@export
@numba.jit(nopython=True, nogil=True, cache=True)
def baseline(records, baseline_samples=40, flip=True,
             allow_sloppy_chunking=False, fallback_baseline=16000):
    """Determine baseline as the average of the first baseline_samples
    of each pulse. Subtract the pulse data from int(baseline),
    and store the baseline mean and rms.

    :param baseline_samples: number of samples at start of pulse to average
    to determine the baseline.
    :param flip: If true, flip sign of data
    :param allow_sloppy_chunking: Allow use of the fallback_baseline in case
    the 0th fragment of a pulse is missing
    :param fallback_baseline: Fallback baseline (ADC counts)

    Assumes records are sorted in time (or at least by channel, then time).

    Assumes record_i information is accurate -- so don't cut pulses before
    baselining them!
    """
    if not len(records):
        return records

    # Array for looking up last baseline (mean, rms) seen in channel
    # We only care about the channels in this set of records; a single .max()
    # is worth avoiding the hassle of passing n_channels around
    n_channels = records['channel'].max() + 1
    last_bl_in = np.zeros((n_channels, 2), dtype=np.float32)
    seen_first = np.zeros(n_channels, dtype=np.bool_)

    for d_i, d in enumerate(records):

        # Compute the baseline if we're the first record of the pulse,
        # otherwise take the last baseline we've seen in the channel
        if d['record_i'] == 0:
            seen_first[d['channel']] = True
            w = d['data'][:baseline_samples]
            last_bl_in[d['channel']] = bl, rms = w.mean(), w.std()
        else:
            bl, rms = last_bl_in[d['channel']]
            if not seen_first[d['channel']]:
                if not allow_sloppy_chunking:
                    print(d.time, d.channel, d.record_i)
                    raise RuntimeError("Cannot baseline, missing 0th fragment!")
                bl = last_bl_in[d['channel']] = fallback_baseline
                rms = np.nan

        # Subtract baseline from all data samples in the record
        # (any additional zeros should be kept at zero)
        d['data'][:d['length']] = (
            (-1 if flip else 1) * (d['data'][:d['length']] - int(bl)))
        d['baseline'] = bl
        d['baseline_rms'] = rms


@export
def raw_to_records(raw_records):
    records = np.zeros(
        len(raw_records),
        dtype=strax.record_dtype(
            record_length_from_dtype(raw_records.dtype)))
    strax.copy_to_buffer(raw_records, records, '_copy_raw_records')
    return records


@export
def copy_raw_records(old, new):
    warn('Deprecated, use strax.copy_to_buffer')
    strax.copy_to_buffer(old, new, '_copy_raw_records')


@export
def record_length_from_dtype(dtype):
    return len(np.zeros(1, dtype)[0]['data'])


@export
@numba.jit(nopython=True, nogil=True, cache=True)
def zero_out_of_bounds(records):
    """"Set waveforms to zero out of pulse bounds
    """
    if not len(records):
        return records
    samples_per_record = len(records[0]['data'])

    for r in records:
        if r['length'] < samples_per_record:
            r['data'][r['length']:] = 0


@export
@numba.jit(nopython=True, nogil=True, cache=True)
def integrate(records):
    """Integrate records in-place"""
    if not len(records):
        return
    for i, r in enumerate(records):
        records[i]['area'] = (
            r['data'].sum() * 2**r['amplitude_bit_shift']
            # Add floating part of baseline * number of samples
            # int(round()) the result since the area field is an int
            + int(round((r['baseline'] % 1) * r['length'])))


@export
@numba.jit(nopython=True, nogil=True, cache=True)
def record_links(records):
    """Return (prev_r, next_r), each arrays of indices of previous/next
    record in the same pulse, or -1 if this is not applicable
    """
    # TODO: needs tests
    if not len(records):
        return (
            np.ones(0, dtype=np.int32) * NO_RECORD_LINK,
            np.ones(0, dtype=np.int32) * NO_RECORD_LINK)
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


@export
def find_hits(records,
              min_amplitude: ty.Union[int, np.ndarray] = 15,
              min_height_over_noise: ty.Union[int, np.ndarray] = 0):
    """Return hits (intervals >= threshold) found in records.
    Hits that straddle record boundaries are split (TODO: fix this?)

    NB: returned hits are NOT sorted yet!
    """
    if not len(records):
        return np.zeros(0, dtype=strax.hit_dtype)
    if isinstance(min_amplitude, (tuple, list)):
        min_amplitude = np.array(min_amplitude)
    if isinstance(min_height_over_noise, (tuple, list)):
        min_height_over_noise = np.array(min_height_over_noise)

    # Convert to per-channel thresholds if needed
    amp_per_ch = isinstance(min_amplitude, np.ndarray)
    hon_per_ch = isinstance(min_height_over_noise, np.ndarray)
    if not (amp_per_ch and hon_per_ch):
        # At least one of the thresholds was specified as a number
        # (constant over all channels)
        # First infer the number of channels
        if amp_per_ch:
            n_channels = len(min_amplitude)
        elif hon_per_ch:
            n_channels = len(min_height_over_noise)
        else:
            n_channels = records['channel'].max() + 1

        # Then create constant arrays for arguments we don't have
        # per-channel thresholds for yet
        if not amp_per_ch:
            min_amplitude = min_amplitude * np.ones(n_channels)
        if not hon_per_ch:
            min_height_over_noise = min_height_over_noise * np.ones(n_channels)

    # Do the actual hitfinding
    return _find_hits(records, min_amplitude, min_height_over_noise)


# Chunk size should be at least a thousand,
# else copying buffers / switching context dominates over actual computation
# No max_duration argument: hits terminate at record boundaries, and
# anyone insane enough to try O(sec) long records deserves to be punished
# TODO: this ignores amplitude_bit_shift, since this function also
# has to support raw_records, which don't have that field.
@strax.growing_result(strax.hit_dtype, chunk_size=int(1e4))
@numba.njit(nogil=True, cache=True)
def _find_hits(records, min_amplitude, min_height_over_noise,
               _result_buffer=None):
    buffer = _result_buffer
    if not len(records):
        return
    offset = 0
    n_channels = len(min_amplitude)

    for record_i, r in enumerate(records):
        # print("Starting record ', record_i)
        in_interval = False
        hit_start = -1
        if r['channel'] >= n_channels:
            print(r['channel'], n_channels)
            raise ValueError("Too few channel thresholds specified")

        area = height = 0
        threshold = max(
            min_amplitude[r['channel']],
            r['baseline_rms'] * min_height_over_noise[r['channel']])
        n_samples = r['length']

        # If someone passes a length > n_samples record,
        # and we don't abort here, numba will happily overrun the buffer!
        assert n_samples <= len(r['data'])

        for i in range(n_samples):
            # We can't use enumerate over r['data'],
            # numba gives errors if we do.
            # maybe file an issue?
            x = r['data'][i]

            satisfy_threshold = x >= threshold
            # print(x, satisfy_threshold, in_interval, hit_start)

            if not in_interval and satisfy_threshold:
                # Start of a hit
                in_interval = True
                hit_start = i
                height = max(x, height)

            if in_interval:
                if not satisfy_threshold:
                    # Hit ends at the start of this sample
                    hit_end = i
                    in_interval = False
                else:
                    area += x
                    height = max(height, x)

                    if i == n_samples - 1:
                        # Hit ends at the *end* of this sample
                        # (because the record ends)
                        hit_end = i + 1
                        in_interval = False

                if not in_interval:
                    # print('saving hit')
                    # Hit is done, add it to the result
                    if hit_end == hit_start:
                        print(r['time'], r['channel'], hit_start)
                        raise ValueError(
                            "Caught attempt to save zero-length hit!")
                    res = buffer[offset]

                    res['threshold'] = threshold
                    res['left'] = hit_start
                    res['right'] = hit_end
                    res['time'] = r['time'] + hit_start * r['dt']
                    # Note right bound is exclusive, no + 1 here:
                    res['length'] = hit_end - hit_start
                    res['dt'] = r['dt']
                    res['channel'] = r['channel']
                    res['record_i'] = record_i

                    # Store areas and height.
                    baseline_fpart = r['baseline'] % 1
                    area += res['length'] * baseline_fpart
                    res['area'] = area
                    res['height'] = height + baseline_fpart
                    area = height = 0

                    # Yield buffer to caller if needed
                    offset += 1
                    if offset == len(buffer):
                        yield offset
                        offset = 0

                    # Clear stuff, just for easier debugging
                    # hit_start = 0
                    # hit_end = 0
    yield offset


@numba.njit(cache=True, nogil=True)
def _waveforms_to_float(wv, bl):
    """Convert waveforms to float and restore baseline"""
    return wv.astype(np.float32) + (bl % 1).reshape(-1, 1)


@export
def filter_records(r, ir):
    """Apply filter with impulse response ir over the records r.
    Assumes the filter origin is at the impulse response maximum.

    :param ws: Waveform matrix, must be float
    :param ir: Impulse response, must have odd length. Will normalize.
    :param prev_r: Previous record map from strax.record_links
    :param next_r: Next record map from strax.record_links
    """
    if not len(r):
        return r
    ws = _waveforms_to_float(r['data'], r['baseline'])

    prev_r, next_r = strax.record_links(r)
    ws_filtered = filter_waveforms(
        ws,
        (ir / ir.sum()).astype(np.float32),
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
