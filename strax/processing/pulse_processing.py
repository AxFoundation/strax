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


# -----------
# Default thresholds for find_hits:
# -----------
default_thresholds = np.zeros(248,
                              dtype=[(('Hitfinder threshold in absolute adc counts above baseline',
                                       'absolute_adc_counts_threshold'), np.int16),
                                     (('Multiplicator for a RMS based threshold (h_o_n * RMS).', 'height_over_noise'),
                                      np.float32),
                                     (('Channel/PMT number', 'channel'), np.int16)
                                     ])
default_thresholds['absolute_adc_counts_threshold'] = 15
default_thresholds['channel'] = np.arange(0, 248, 1, dtype=np.int16)
# Chunk size should be at least a thousand,
# else copying buffers / switching context dominates over actual computation
# No max_duration argument: hits terminate at record boundaries, and
# anyone insane enough to try O(sec) long records deserves to be punished
@export
@strax.growing_result(strax.hit_dtype, chunk_size=int(1e4))
@numba.jit(nopython=True, nogil=True, cache=True)
def find_hits(records, threshold=default_thresholds, nbaseline=40, _result_buffer=None):
    """Return hits (intervals above threshold) found in records.
    Hits that straddle record boundaries are split (TODO: fix this?)

    NB: returned hits are NOT sorted yet!
    """
    buffer = _result_buffer
    if not len(records):
        return
    samples_per_record = len(records[0]['data'])
    offset = 0

    # Bookkeeping of rms values:
    nchannels = len(threshold['channel'])
    rms_values = np.zeros(nchannels)

    for record_i, r in enumerate(records):
        # print("Starting record ', record_i)
        in_interval = False
        hit_start = -1
        
        # Computing rms:
        if r['record_i'] == 0:
            rms = _baseline_rms(r, nbaseline)
            rms_values[threshold['channel'] == r['channel']] = rms   # channel could not start with 0 e.g. nVETO and
                                                                     # might not be successive/equidistant
        else:
            rms = rms_values[threshold['channel'] == r['channel']][0]
            
        area = height = 0
        
        for i in range(samples_per_record):
            # We can't use enumerate over r['data'],
            # numba gives errors if we do.
            # TODO: file issue?
            x = r['data'][i]

            th = max(threshold['absolute_adc_counts_threshold'][threshold['channel'] == r['channel']][0],
                     rms * threshold['height_over_noise'][threshold['channel'] == r['channel']][0])
            above_threshold = x > th  # can ignore the flat part since > and [0,1)?

            # print(r['data'][i], above_threshold, in_interval, hit_start)

            if not in_interval and above_threshold:
                # Start of a hit
                in_interval = True
                hit_start = i
                height = max(x, height)

            if in_interval:
                if not above_threshold:
                    # Hit ends at the start of this sample
                    hit_end = i
                    in_interval = False
                else:
                    area += x
                    height = max(height, x)

                    if i == samples_per_record - 1:
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

                    res['rms'] = rms
                    res['threshold'] = th  # No idea if we want to keep this after tuning...
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
def _baseline_rms(rr, n_samples=40):
    """
    Function which estimates the baseline rms within a certain number of samples.

    The rms value is estimated for all samples with adc counts <= 0.

    Args:
        rr (raw_records): single raw_record

    Keyword Args:
        n_samples (int): First n samples on which the rms is estimated.
    """
    d = rr['data']
    b = rr['baseline']%1
    d_b = d + b
    n = 0
    rms = 0
    for s in d_b[:n_samples]:
        if s < 0:
            rms += s**2
            n += 1
    # TODO: Ask maybe other fall back solution?
    if n == 0:
        return 42000

    return np.sqrt(rms / n)


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
