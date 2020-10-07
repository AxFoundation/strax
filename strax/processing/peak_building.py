import numpy as np
import numba

import strax
from strax import utils
from strax.dtypes import peak_dtype, DIGITAL_SUM_WAVEFORM_CHANNEL
export, __all__ = strax.exporter()


@export
@utils.growing_result(dtype=peak_dtype(), chunk_size=int(1e4))
@numba.jit(nopython=True, nogil=True, cache=True)
def find_peaks(hits, adc_to_pe,
               gap_threshold=300,
               left_extension=20, right_extension=150,
               min_area=0,
               min_channels=2,
               _result_buffer=None, result_dtype=None):
    """Return peaks made from grouping hits together
    Assumes all hits have the same dt
    :param hits: Hit (or any interval) to group
    :param left_extension: Extend peaks by this many ns left
    :param right_extension: Extend peaks by this many ns right
    :param gap_threshold: No hits for this much ns means new peak
    :param min_channels: Peaks with less contributing channels are not returned
    :param min_area: Peaks with less than min_area are not returned
    """
    buffer = _result_buffer
    offset = 0
    if not len(hits):
        return
    assert hits[0]['dt'] > 0, "Hit does not indicate sampling time"
    assert min_channels >= 1, "min_channels must be >= 1"
    assert gap_threshold > left_extension + right_extension, \
        "gap_threshold must be larger than left + right extension"
    # If you write it like below, you get integer wraparound errors
    # TODO :-( File numba issue?
    # assert max_duration < np.iinfo(np.int32).max * hits[0]['dt'], \
    #   "Max duration must fit in a 32-bit signed integer"

    n_channels = len(buffer[0]['area_per_channel'])
    area_per_channel = np.zeros(n_channels, dtype=np.float32)

    in_peak = False
    peak_endtime = 0
    for hit_i, hit in enumerate(hits):
        p = buffer[offset]
        t0 = hit['time']
        dt = hit['dt']
        t1 = hit['time'] + dt * hit['length']

        if in_peak:
            # This hit continues an existing peak
            p['max_gap'] = max(p['max_gap'], t0 - peak_endtime)

        else:
            # This hit starts a new peak candidate
            area_per_channel *= 0
            peak_endtime = t1
            p['time'] = t0 - left_extension
            p['channel'] = DIGITAL_SUM_WAVEFORM_CHANNEL
            p['dt'] = dt
            # These are necessary as prev peak may have been rejected:
            p['n_hits'] = 0
            p['area'] = 0
            in_peak = True
            p['max_gap'] = 0

        # Add hit's properties to the current peak candidate
        p['n_hits'] += 1
        peak_endtime = max(peak_endtime, t1)
        hit_area_pe = hit['area'] * adc_to_pe[hit['channel']]
        area_per_channel[hit['channel']] += hit_area_pe
        p['area'] += hit_area_pe

        # Look at the next hit to see if THIS hit is the last in a peak.
        # If this is the final hit, it is last by definition.
        if (hit_i == len(hits) - 1
                or hits[hit_i+1]['time'] - peak_endtime >= gap_threshold):
            # Next hit (if it exists) will initialize the new peak candidate
            in_peak = False

            # Do not save if tests are not met. Next hit will erase temp info
            if p['area'] < min_area:
                continue
            n_channels = (area_per_channel != 0).sum()
            if n_channels < min_channels:
                continue

            # Compute final quantities
            p['length'] = (peak_endtime - p['time'] + right_extension) / dt
            if p['length'] <= 0:
                raise ValueError(
                    "Caught attempt to save nonpositive peak length?!")
            p['area_per_channel'][:] = area_per_channel

            # Save the current peak, advance the buffer
            offset += 1
            if offset == len(buffer):
                yield offset
                offset = 0

    yield offset


@export
@numba.jit(nopython=True, nogil=True, cache=True)
def store_downsampled_waveform(p, wv_buffer):
    """Downsample the waveform in buffer and store it in p['data']

    :param p: Row of a strax peak array, or compatible type.
    Note that p['dt'] is adjusted to match the downsampling.
    :param wv_buffer: numpy array containing sum waveform during the peak
    at the input peak's sampling resolution p['dt'].

    The number of samples to take from wv_buffer, and thus the downsampling
    factor, is determined from p['dt'] and p['length'].

    When downsampling results in a fractional number of samples, the peak is
    shortened rather than extended. This causes data loss, but it is
    necessary to prevent overlaps between peaks.
    """
    n_samples = len(p['data'])
    downsample_factor = int(np.ceil(p['length'] / n_samples))
    if downsample_factor > 1:
        # Compute peak length after downsampling.
        # Do not ceil: see docstring!
        p['length'] = int(np.floor(p['length'] / downsample_factor))
        p['data'][:p['length']] = \
            wv_buffer[:p['length'] * downsample_factor] \
                .reshape(-1, downsample_factor) \
                .sum(axis=1)
        p['dt'] *= downsample_factor
    else:
        p['data'][:p['length']] = wv_buffer[:p['length']]


@export
@numba.jit(nopython=True, nogil=True, cache=True)
def sum_waveform(peaks, records, adc_to_pe, select_peaks_indices=None):
    """Compute sum waveforms for all peaks in peaks
    Will downsample sum waveforms if they do not fit in per-peak buffer

    :arg select_peaks_indices: Indices of the peaks for partial
    processing. In the form of np.array([np.int, np.int, ..]). If
    None (default), all the peaks are used for the summation.

    Assumes all peaks AND pulses have the same dt!
    """
    if not len(records):
        return
    if not len(peaks):
        return
    if select_peaks_indices is None:
        select_peaks_indices = np.arange(len(peaks))
    if not len(select_peaks_indices):
        return
    dt = records[0]['dt']

    # Big buffer to hold even largest sum waveforms
    # Need a little more even for downsampling..
    swv_buffer = np.zeros(peaks['length'].max() * 2, dtype=np.float32)

    # Index of first record that could still contribute to subsequent peaks
    # Records before this do not need to be considered anymore
    left_r_i = 0

    n_channels = len(peaks[0]['area_per_channel'])
    area_per_channel = np.zeros(n_channels, dtype=np.float32)

    for peak_i in select_peaks_indices:
        p = peaks[peak_i]
        # Clear the relevant part of the swv buffer for use
        # (we clear a bit extra for use in downsampling)
        p_length = p['length']
        swv_buffer[:min(2 * p_length, len(swv_buffer))] = 0

        # Clear area and area per channel
        # (in case find_peaks already populated them)
        area_per_channel *= 0
        p['area'] = 0

        # Find first record that contributes to this peak
        for left_r_i in range(left_r_i, len(records)):
            r = records[left_r_i]
            # TODO: need test that fails if we replace < with <= here
            if p['time'] < r['time'] + r['length'] * dt:
                break
        else:
            # Records exhausted before peaks exhausted
            # TODO: this is a strange case, maybe raise warning/error?
            break

        # Scan over records that overlap
        for right_r_i in range(left_r_i, len(records)):
            r = records[right_r_i]
            ch = r['channel']
            multiplier = 2**r['amplitude_bit_shift']
            assert p['dt'] == r['dt'], "Records and peaks must have same dt"

            shift = (p['time'] - r['time']) // dt
            n_r = r['length']
            n_p = p_length

            if shift <= -n_p:
                # Record is completely to the right of the peak;
                # we've seen all overlapping records
                break

            if n_r <= shift:
                # The (real) data in this record does not actually overlap
                # with the peak
                # (although a previous, longer record did overlap)
                continue

            (r_start, r_end), (p_start, p_end) = strax.overlap_indices(
                r['time'] // dt, n_r,
                p['time'] // dt, n_p)

            max_in_record = r['data'][r_start:r_end].max() * multiplier
            p['saturated_channel'][ch] |= np.int8(max_in_record >= r['baseline'])

            bl_fpart = r['baseline'] % 1
            # TODO: check numba does casting correctly here!
            pe_waveform = adc_to_pe[ch] * (
                    multiplier * r['data'][r_start:r_end]
                    + bl_fpart)

            swv_buffer[p_start:p_end] += pe_waveform

            area_pe = pe_waveform.sum()
            area_per_channel[ch] += area_pe
            p['area'] += area_pe

        store_downsampled_waveform(p, swv_buffer)

        p['n_saturated_channels'] = p['saturated_channel'].sum()
        p['area_per_channel'][:] = area_per_channel


@export
def find_peak_groups(peaks, gap_threshold,
                     left_extension=0, right_extension=0):
    """Return boundaries of groups of peaks separated by gap_threshold,
    extended left and right.

    :param peaks: Peaks to group
    :param gap_threshold: Minimum gap between peaks
    :param left_extension: Extend groups by this many ns left
    :param right_extension: " " right
    :return: time, endtime arrays of group boundaries
    """
    # Mock up a "hits" array so we can just use the existing peakfinder
    # It doesn't work on raw peaks, since they might have different dts
    # TODO: is there no cleaner way?
    fake_hits = np.zeros(len(peaks), dtype=strax.hit_dtype)
    fake_hits['dt'] = 1
    fake_hits['area'] = 1
    fake_hits['time'] = peaks['time']
    # TODO: could this cause int overrun nonsense anywhere?
    fake_hits['length'] = strax.endtime(peaks) - peaks['time']
    fake_peaks = strax.find_peaks(
        fake_hits, adc_to_pe=np.ones(1),
        gap_threshold=gap_threshold,
        left_extension=left_extension, right_extension=right_extension,
        min_channels=1, min_area=0)
    return fake_peaks['time'], strax.endtime(fake_peaks)


##
# Lone hit integration
##

@numba.njit(nogil=True, cache=True)
def _find_hit_integration_bounds(
        lone_hits, peaks, records, save_outside_hits, n_channels):
    """"Update lone hits to include integration bounds

    save_outside_hits: in ns!!
    """
    result = np.zeros((len(lone_hits), 2), dtype=np.int64)
    if not len(lone_hits):
        return result

    # By default, use save_outside_hits to determine bounds
    result[:, 0] = lone_hits['time'] - save_outside_hits[0]
    result[:, 1] = strax.endtime(lone_hits) + save_outside_hits[1]

    NO_EARLIER_HIT = -1
    last_hit_index = np.ones(n_channels, dtype=np.int32) * NO_EARLIER_HIT

    n_peaks = len(peaks)
    FAR_AWAY = 9223372036_854775807   # np.iinfo(np.int64).max, April 2262
    peak_i = 0

    for hit_i, h in enumerate(lone_hits):
        ch = h['channel']

        # Find end of previous peak and start of next peak
        # (note peaks are disjoint from any lone hit, even though
        # lone hits may not be disjoint from each other)
        while peak_i < n_peaks and peaks[peak_i]['time'] < h['time']:
            peak_i += 1
        prev_p_end = strax.endtime(peaks[peak_i - 1]) if peak_i != 0 else 0
        next_p_start = peaks[peak_i]['time'] if peak_i != n_peaks else FAR_AWAY


        # Ensure we do not integrate parts of peaks
        # or (at least for now) beyond the record in which the hit was found
        r = records[h['record_i']]
        result[hit_i][0] = max(prev_p_end,
                               r['time'],
                               result[hit_i][0])
        result[hit_i][1] = min(next_p_start,
                               strax.endtime(r),
                               result[hit_i][1])

        if last_hit_index[ch] != NO_EARLIER_HIT:
            # Ensure previous hit does not integrate the over-threshold region
            # of this hit
            result[last_hit_index[ch]][1] = min(result[last_hit_index[ch]][1],
                                                h['time'])
            # Ensure this hit doesn't integrate anything the previous hit
            # already integrated
            result[hit_i][0] = max(result[last_hit_index[ch]][1],
                                   result[hit_i][0])

        last_hit_index[ch] = hit_i

    # Convert to index in record and store
    t0 = records[lone_hits['record_i']]['time']
    dt = records[lone_hits['record_i']]['dt']
    for hit_i, h in enumerate(lone_hits):
        h['left_integration'] = (result[hit_i, 0] - t0[hit_i]) // dt[hit_i]
        h['right_integration'] = (result[hit_i, 1] - t0[hit_i]) // dt[hit_i]


@export
@numba.njit(nogil=True, cache=True)
def integrate_lone_hits(
        lone_hits, records, peaks, save_outside_hits, n_channels):
    """Update the area of lone_hits to the integral in ADCcounts x samples

    :param lone_hits: Hits outside of peaks
    :param records: Records in which hits and peaks were found
    :param peaks: Peaks
    :param save_outside_hits: (left, right) *TIME* with wich we should extend
    the integration window of hits
    the integration region
    :param n_channels: number of channels

    TODO: this doesn't extend the integration range beyond record boundaries
    """
    _find_hit_integration_bounds(
        lone_hits, peaks, records, save_outside_hits, n_channels)
    for hit_i, h in enumerate(lone_hits):
        r = records[h['record_i']]
        start, end = h['left_integration'], h['right_integration']
        # TODO: when we add amplitude multiplier, adjust this too!
        h['area'] = (
                r['data'][start:end].sum()
                + (r['baseline'] % 1) * (end - start))
