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

        if not in_peak:
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
def sum_waveform(peaks, records, adc_to_pe, n_channels=248):
    """Compute sum waveforms for all peaks in peaks
    Will downsample sum waveforms if they do not fit in per-peak buffer

    :param n_channels: Number of channels that contribute to the total area
    and n_saturated_channels.
    For further channels we still calculate area_per_channel and saturated_channel.

    Assumes all peaks and pulses have the same dt
    """
    if not len(records):
        return
    if not len(peaks):
        return
    dt = records[0]['dt']
    sum_wv_samples = len(peaks[0]['data'])

    # Big buffer to hold even largest sum waveforms
    # Need a little more even for downsampling..
    swv_buffer = np.zeros(peaks['length'].max() * 2, dtype=np.float32)

    # Index of first record that could still contribute to subsequent peaks
    # Records before this do not need to be considered anymore
    left_r_i = 0

    n_channels = len(peaks[0]['area_per_channel'])
    area_per_channel = np.zeros(n_channels, dtype=np.float32)

    for peak_i, p in enumerate(peaks):
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

            # Range of record that contributes to peak
            r_start = max(0, shift)
            r_end = min(n_r, shift + n_p)
            assert r_end > r_start

            # Range of peak that receives record
            p_start = max(0, -shift)
            p_end = min(n_p, -shift + n_r)

            assert p_end - p_start == r_end - r_start, "Ouch, off-by-one error"

            max_in_record = r['data'][r_start:r_end].max()
            p['saturated_channel'][ch] = int(max_in_record >= r['baseline'])

            bl_fpart = r['baseline'] % 1
            # TODO: check numba does casting correctly here!
            pe_waveform = adc_to_pe[ch] * (
                    r['data'][r_start:r_end] + bl_fpart)

            swv_buffer[p_start:p_end] += pe_waveform

            area_pe = pe_waveform.sum()
            area_per_channel[ch] += area_pe
            p['area'] += area_pe

        # Store the sum waveform
        # Do we need to downsample the swv to store it?
        downs_f = int(np.ceil(p_length / sum_wv_samples))
        if downs_f > 1:
            # Compute peak length after downsampling.
            # We floor rather than ceil here, potentially cutting off
            # some samples from the right edge of the peak.
            # If we would ceil, the peak could grow larger and
            # overlap with a subsequent next peak, crashing strax later.
            new_ns = p['length'] = int(np.floor(p_length / downs_f))
            p['data'][:new_ns] = \
                swv_buffer[:new_ns * downs_f].reshape(-1, downs_f).sum(axis=1)
            p['dt'] *= downs_f
        else:
            p['data'][:p_length] = swv_buffer[:p_length]

        # Store the saturation count and area per channel
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