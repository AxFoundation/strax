import numpy as np
import numba

from strax import utils
from strax.dtypes import peak_dtype, DIGITAL_SUM_WAVEFORM_CHANNEL

__all__ = 'find_peaks sum_waveform'.split()


@utils.growing_result(dtype=peak_dtype(), chunk_size=int(1e4))
@numba.jit(nopython=True, nogil=True, cache=True)
def find_peaks(hits, to_pe,
               gap_threshold=300,
               left_extension=20, right_extension=150,
               min_hits=3, min_area=0,
               max_duration=int(1e9),
               _result_buffer=None, result_dtype=None):
    """Return peaks made from grouping hits together
    Assumes all hits have the same dt
    :param hits: Hit (or any interval) to group
    :param left_extension: Extend peaks by this many ns left
    :param right_extension: Extend peaks by this many ns right
    :param gap_threshold: No hits for this much ns means new peak
    :param min_hits: Peaks with less than min_hits are not returned
    :param min_area: Peaks with less than min_area are not returned
    :param max_duration: Peaks are forcefully ended after this many ns
    """
    buffer = _result_buffer
    offset = 0
    if not len(hits):
        return
    assert hits[0]['dt'] > 0, "Hit does not indicate sampling time"
    assert min_hits > 0, "min_hits must be > 1"
    assert gap_threshold > left_extension + right_extension, \
        "gap_threshold must be larger than left + right extension"
    assert max_duration / hits[0]['dt'] < np.iinfo(np.int32).max, \
        "Max duration must fit in a 32-bit signed integer"
    # If you write it like below, you get integer wraparound errors
    # TODO :-( File numba issue?
    # assert max_duration < np.iinfo(np.int32).max * hits[0]['dt'], \
    #   "Max duration must fit in a 32-bit signed integer"

    area_per_channel = np.zeros(len(buffer[0]['area_per_channel']),
                                dtype=np.int32)
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
        p['area_per_channel'][hit['channel']] += hit['area']
        p['area'] += hit['area'] * to_pe[hit['channel']]

        # Look at the next hit to see if THIS hit is the last in a peak.
        # If this is the final hit, it is last by definition.
        if (hit_i == len(hits) - 1
                or hits[hit_i+1]['time'] - peak_endtime >= gap_threshold):
            # Next hit (if it exists) will initialize the new peak candidate
            in_peak = False

            # Do not save if tests are not met. Next hit will erase temp info
            if not (p['n_hits'] >= min_hits and p['area'] >= min_area):
                continue

            # Compute final quantities
            p['length'] = (peak_endtime - p['time'] + right_extension) / dt
            p['area_per_channel'][:] = area_per_channel

            # Save the current peak, advance the buffer
            offset += 1
            if offset == len(buffer):
                yield offset
                offset = 0

    yield offset


@numba.jit(nopython=True, nogil=True, cache=True)
def sum_waveform(peaks, records, adc_to_pe):
    """Compute sum waveforms for all peaks in peaks
    Will downsample sum waveforms if they do not fit in per-peak buffer

    Assumes all peaks and pulses have the same dt
    """
    if not len(records):
        return
    samples_per_record = len(records[0]['data'])
    dt = records[0]['dt']
    time_per_record = samples_per_record * dt
    sum_wv_samples = len(peaks[0]['data'])

    # Big buffer to hold even largest sum waveforms
    # Need a little more even for downsampling..
    swv_buffer = np.zeros(peaks['length'].max() * 2, dtype=np.float32)

    # Index to a window of records
    left_r_i = 0

    for peak_i, p in enumerate(peaks):
        # print("Peak ", peak_i)
        # Clear the relevant part of the swv buffer for use
        # (we clear a bit extra for use in downsampling)
        p_length = p['length']
        swv_buffer[:min(2 * p_length, len(swv_buffer))] = 0

        # Find first record that could contribute to peak
        for left_r_i in range(left_r_i, len(records)):
            if records[left_r_i]['time'] + time_per_record >= p['time']:
                break

        if left_r_i == len(records):
            # No record contributes to the last peak!
            break

        # Scan ahead over records that contribute
        for right_r_i in range(left_r_i, len(records)):
            r = records[right_r_i]
            ch = r['channel']
            # print("record ", right_r_i, ' [', r['time'], ',',
            # r['time'] + r['length'], ')')

            s = int((p['time'] - r['time']) // dt)
            n_r = samples_per_record
            n_p = p_length

            if s < -n_p:
                # print("out of range! s is ", s)
                # Record is fully out of range
                break

            # Range of record that contributes to peak
            r_start = max(0, s)
            r_end = min(n_r, s + n_p)

            # TODO Do we need .astype(np.int32).sum() ??
            p['area_per_channel'][ch] += r['data'][r_start:r_end].sum()

            # Range of peak that receives record
            p_start = max(0, -s)
            p_end = min(n_p, -s + n_r)

            assert p_end - p_start == r_end - r_start, "Ouch, off-by-one error"

            # print("contributes ", r_start, " to ", r_end)
            # print("insert in ", p_start, " to ", p_end)
            if p_end - p_start > 0:
                swv_buffer[p_start:p_end] += \
                    r['data'][r_start:r_end] * adc_to_pe[ch]

        # Store the sum waveform
        # Do we need to downsample the swv to store it?
        downs_f = int(np.ceil(p_length / sum_wv_samples))
        if downs_f > 1:
            # New number of samples in the peak
            new_ns = p['length'] = int(np.ceil(p_length / downs_f))
            p['data'][:new_ns] = \
                swv_buffer[:new_ns * downs_f].reshape(-1, downs_f).sum(axis=1)
            p['dt'] *= downs_f
        else:
            p['data'][:p_length] = swv_buffer[:p_length]

        # Store the total area
        p['area'] = (p['area_per_channel'] * adc_to_pe).sum()
