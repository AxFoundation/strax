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
               max_duration=10_000_000,
               _result_buffer=None, result_dtype=None):
    """Return peaks made from grouping hits together
    Assumes all hits have the same dt
    :param hits: Hit (or any interval) to group
    :param left_extension: Extend peaks by this many ns left
    :param right_extension: Extend peaks by this many ns right
    :param gap_threshold: No hits for this much ns means new peak
    :param min_area: Peaks with less than min_area are not returned
    :param min_channels: Peaks with less contributing channels are not returned
    :param max_duration: max duration time of merged peak in ns
    """
    buffer = _result_buffer
    offset = 0
    if not len(hits):
        return
    assert hits[0]['dt'] > 0, "Hit does not indicate sampling time"
    assert min_channels >= 1, "min_channels must be >= 1"
    assert gap_threshold > left_extension + right_extension, \
        "gap_threshold must be larger than left + right extension"
    assert max(hits['channel']) < len(adc_to_pe), "more channels than to_pe"
    # Magic number comes from
    #   np.iinfo(p['dt'].dtype).max*np.shape(p['data'])[1] = 429496729400 ns
    # but numba does not like it
    assert left_extension+max_duration+right_extension < 429496729400, (
        "Too large max duration causes integer overflow")

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
        # Finally, make sure that if we include the next hit, we are not
        # exceeding the max_duration.
        is_last_hit = hit_i == len(hits) - 1
        peak_too_long = next_hit_is_far = False
        if not is_last_hit:
            # These can only be computed if there is a next hit
            next_hit = hits[hit_i + 1]
            next_hit_is_far = next_hit['time'] - peak_endtime >= gap_threshold
            # Peaks may not extend the max_duration
            peak_too_long = (next_hit['time'] - p['time']
                             + next_hit['dt'] * next_hit['length']
                             + left_extension
                             + right_extension) > max_duration
        if is_last_hit or next_hit_is_far or peak_too_long:
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
                # This is most likely caused by a negative dt
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
def sum_waveform(peaks, hits, records, record_links, adc_to_pe, select_peaks_indices=None):
    """Compute sum waveforms for all peaks in peaks. Only builds summed
    waveform other regions in which hits were found. This is required
    to avoid any bias due to zero-padding and baselining.
    Will downsample sum waveforms if they do not fit in per-peak buffer

    :param peaks: Peaks for which the summed waveform should be build.
    :param hits: Hits which are inside peaks. Must be sorted according
        to record_i.
    :param records: Records to be used to build peaks.
    :param record_links: Tuple of previous and next records.
    :param select_peaks_indices: Indices of the peaks for partial
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
    n_samples_record = len(records[0]['data'])
    prev_record_i, next_record_i = record_links

    # Big buffer to hold even largest sum waveforms
    # Need a little more even for downsampling..
    swv_buffer = np.zeros(peaks['length'].max() * 2, dtype=np.float32)

    n_channels = len(peaks[0]['area_per_channel'])
    area_per_channel = np.zeros(n_channels, dtype=np.float32)

    # Hit index for hits in peaks
    left_h_i = 0
    # Create hit waveform buffer
    hit_waveform = np.zeros(hits['length'].max(), dtype=np.float32)

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

        # Find first hit that contributes to this peak
        for left_h_i in range(left_h_i, len(hits)):
            h = hits[left_h_i]
            # TODO: need test that fails if we replace < with <= here
            if p['time'] < h['time'] + h['length'] * dt:
                break
        else:
            # Hits exhausted before peaks exhausted
            # TODO: this is a strange case, maybe raise warning/error?
            break

        # Scan over hits that overlap with peak
        for right_h_i in range(left_h_i, len(hits)):
            h = hits[right_h_i]
            record_i = h['record_i']
            ch = h['channel']
            assert p['dt'] == h['dt'], "Hits and peaks must have same dt"

            shift = (p['time'] - h['time']) // dt
            n_samples_hit = h['length']
            n_samples_peak = p_length

            if shift <= -n_samples_peak:
                # Hit is completely to the right of the peak;
                # we've seen all overlapping records
                break

            if n_samples_hit <= shift:
                # The (real) data in this record does not actually overlap
                # with the peak
                # (although a previous, longer hit did overlap)
                continue

            # Get overlapping samples between hit and peak:
            (h_start, h_end), (p_start, p_end) = strax.overlap_indices(
                h['time'] // dt, n_samples_hit,
                p['time'] // dt, n_samples_peak)

            hit_waveform[:] = 0

            # Get record which belongs to main part of hit (wo integration bounds):
            r = records[record_i]

            is_saturated = _build_hit_waveform(h, r, hit_waveform)

            # Now check if we also have to go to prev/next record due to integration bounds.
            # If bounds are outside of peak we chop when building the summed waveform later.
            if h['left_integration'] < 0 and prev_record_i[record_i] != -1:
                r = records[prev_record_i[record_i]]
                is_saturated |= _build_hit_waveform(h, r, hit_waveform)

            if h['right_integration'] > n_samples_record and next_record_i[record_i] != -1:
                r = records[next_record_i[record_i]]
                is_saturated |= _build_hit_waveform(h, r, hit_waveform)

            p['saturated_channel'][ch] |= is_saturated

            hit_data = hit_waveform[h_start:h_end]
            hit_data *= adc_to_pe[ch]
            swv_buffer[p_start:p_end] += hit_data

            area_pe = hit_data.sum()
            area_per_channel[ch] += area_pe
            p['area'] += area_pe

        store_downsampled_waveform(p, swv_buffer)

        p['n_saturated_channels'] = p['saturated_channel'].sum()
        p['area_per_channel'][:] = area_per_channel


@numba.njit(cache=True, nogil=True)
def _build_hit_waveform(hit, record, hit_waveform):
    """
    Adds information for overlapping record and hit to hit_waveform.
    Updates hit_waveform inplace. Result is still in ADC counts.

    :returns: Boolean if record saturated within the hit.
    """
    (h_start_record, h_end_record), (r_start, r_end) = strax.overlap_indices(
        hit['time'] // hit['dt'], hit['length'],
        record['time'] // record['dt'], record['length'])

    # Get record properties:
    record_data = record['data'][r_start:r_end]
    multiplier = 2**record['amplitude_bit_shift']
    bl_fpart = record['baseline'] % 1
    max_in_record = record_data.max() * multiplier

    # Build hit waveform:
    hit_waveform[h_start_record:h_end_record] = (multiplier * record_data + bl_fpart)

    return np.int8(max_in_record >= np.int16(record['baseline']))


@export
def find_peak_groups(peaks, gap_threshold,
                     left_extension=0, right_extension=0,
                     max_duration=int(1e9),
                     ):
    """Return boundaries of groups of peaks separated by gap_threshold,
    extended left and right.

    :param peaks: Peaks to group
    :param gap_threshold: Minimum gap between peaks
    :param left_extension: Extend groups by this many ns left
    :param right_extension: " " right
    :param max_duration: max duration time of merged peak in ns
    :return: time, endtime arrays of group boundaries
    """
    # Mock up a "hits" array so we can just use the existing peakfinder
    # It doesn't work on raw peaks, since they might have different dts
    # Maybe there is a cleaner way?
    fake_hits = np.zeros(len(peaks), dtype=strax.hit_dtype)
    fake_hits['dt'] = 1
    fake_hits['area'] = 1
    fake_hits['time'] = peaks['time']
    fake_hits['length'] = strax.endtime(peaks) - peaks['time']
    # Probably int overflow
    assert np.all(fake_hits['length'] > 0), "Attempt to create invalid hit"
    fake_peaks = strax.find_peaks(
        fake_hits, adc_to_pe=np.ones(1),
        gap_threshold=gap_threshold,
        left_extension=left_extension, right_extension=right_extension,
        min_channels=1, min_area=0,
        max_duration=max_duration)
    return fake_peaks['time'], strax.endtime(fake_peaks)


##
# Lone hit integration
##
@export
@numba.njit(nogil=True, cache=True)
def find_hit_integration_bounds(
        hits, excluded_intervals, records, save_outside_hits, n_channels,
        allow_bounds_beyond_records=False):
    """"Update (lone) hits to include integration bounds. Please note
    that time and length of the original hit are not changed!

    :param hits: Hits or lone hits which should be extended by
        integration bounds.
    :param excluded_intervals: Regions in which hits should not extend to. E.g. Peaks
        for lone hits. If not needed just put a zero length
        strax.time_fields array.
    :param records: Records in which hits were found.
    :param save_outside_hits: Hit extension to the left and right in ns
        not samples!!
    :param n_channels: Number of channels for given detector.
    :param allow_bounds_beyond_records: If true extend left/
        right_integration beyond record boundaries. E.g. to negative
        samples for left side.
    """
    result = np.zeros((len(hits), 2), dtype=np.int64)
    if not len(hits):
        return result

    # By default, use save_outside_hits to determine bounds
    result[:, 0] = hits['time'] - save_outside_hits[0]
    result[:, 1] = strax.endtime(hits) + save_outside_hits[1]

    NO_EARLIER_HIT = -1
    last_hit_index = np.ones(n_channels, dtype=np.int32) * NO_EARLIER_HIT

    n_intervals = len(excluded_intervals)
    FAR_AWAY = 9223372036_854775807   # np.iinfo(np.int64).max, April 2262
    interval_i = 0

    for hit_i, h in enumerate(hits):
        ch = h['channel']

        # Find end of previous peak and start of next peak
        # (note peaks are disjoint from any lone hit, even though
        # lone hits may not be disjoint from each other)
        while interval_i < n_intervals and excluded_intervals[interval_i]['time'] < h['time']:
            interval_i += 1

        if interval_i != 0:
            prev_interval_end = strax.endtime(excluded_intervals[interval_i - 1])
        else:
            prev_interval_end = 0

        if interval_i != n_intervals:
            next_interval_start = excluded_intervals[interval_i]['time']
        else:
            next_interval_start = FAR_AWAY

        r = records[h['record_i']]
        if allow_bounds_beyond_records:
            result[hit_i][0] = max(prev_interval_end,
                                   result[hit_i][0])
            result[hit_i][1] = min(next_interval_start,
                                   result[hit_i][1])
        else:
            result[hit_i][0] = max(prev_interval_end,
                                   r['time'],
                                   result[hit_i][0])
            result[hit_i][1] = min(next_interval_start,
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
    t0 = records[hits['record_i']]['time']
    dt = records[hits['record_i']]['dt']
    for hit_i, h in enumerate(hits):
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
    find_hit_integration_bounds(lone_hits, peaks, records, save_outside_hits,
                                n_channels)
    for hit_i, h in enumerate(lone_hits):
        r = records[h['record_i']]
        start, end = h['left_integration'], h['right_integration']
        # TODO: when we add amplitude multiplier, adjust this too!
        h['area'] = (
                r['data'][start:end].sum()
                + (r['baseline'] % 1) * (end - start))
