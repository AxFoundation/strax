import numpy as np
import numba
from strax import utils

from strax.data import peak_dtype

__all__ = 'find_peaks find_large_peaks_roughly sum_waveform'.split()


@utils.growing_result(dtype=peak_dtype(260), chunk_size=int(1e4))
@numba.jit(nopython=True, nogil=True)
def find_peaks(result_buffer, hits,
               left_extension=20,
               right_extension=150,
               gap_threshold=500, min_hits=3, max_duration=int(1e9)):
    """Return peaks made from grouping hits together
    Assumes all hits have the same dt
    :param hits:
    :param left_extension: Extend peaks by this many ns left
    :param right_extension: Extend peaks by this many ns right
    :param gap_threshold: No hits for this much ns means new peak
    :param min_hits: Peaks with less than min_hits are not saved
    :param max_duration: Peaks are forcefully ended after this many ns
    """
    if not len(hits):
        return
    offset = 0
    assert gap_threshold > left_extension + right_extension
    hit_starts = hits['time']
    hit_ends = hit_starts + hits['length'] * hits['dt']

    peak_start = hit_starts[0]
    peak_end = hit_ends[0]
    area = 0
    n_hits = 0

    for i, hit in enumerate(hits[1:]):
        t0 = hit_starts[i]
        t1 = hit_starts[i]
        dt = hit['dt']
        ar = hit['area']

        gap = t0 - peak_end

        if gap > gap_threshold or t0 > peak_start + max_duration:
            # This hit no longer belongs to the same signal
            # store the old signal if it contains enough hits
            if n_hits >= min_hits:
                res = result_buffer[offset]
                res['time'] = peak_start - left_extension
                res['length'] = (peak_end - peak_start + right_extension) / dt
                res['n_hits'] = n_hits
                res['dt'] = dt
                res['area'] = area

                offset += 1
                if offset == len(result_buffer):
                    yield offset
                    offset = 0
            n_hits = 1
            area = ar
            peak_start = t0
            peak_end = t1

        else:
            # Hit continues the current signal
            peak_end = max(t1, peak_end)
            n_hits += 1
            area += ar

    yield offset


@numba.jit(nopython=True, nogil=True)
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

    # Indices to a window of records
    left_r_i = 0
    right_r_i = 0

    for peak_i, p in enumerate(peaks):
        # Clear the relevant part of the swv buffer for use
        # (we clear a bit extra for use in downsampling)
        p_length = p['length']
        swv_buffer[:min(2 * p_length, len(swv_buffer))] = 0

        # Find first record that contributes to peak
        while records[left_r_i]['time'] + time_per_record <= p['time']:
            left_r_i += 1

        # Scan ahead over records that contribute
        right_r_i = left_r_i
        while True:
            r = records[right_r_i]
            ch = r['channel']

            s = int((p['time'] - r['time']) // dt)
            n_r = samples_per_record
            n_p = p_length

            if s < -n_p:
                # Record is fully out of range
                break

            # Range of record that contributes to peak
            r_start = max(0, s)
            r_end = min(n_r, s + n_p)
            # TODO Do we need .astype(np.int32).sum() ??
            # p['area_per_channel'][ch] += r['data'][r_start:r_end].sum()
            p['area_per_channel'][ch] += r['data'][r_start:r_end].sum()

            # Range of peak that receives record
            p_start = max(0, -s)
            p_end = min(n_p, -s + n_r)

            assert p_end - p_start == r_end - r_start, "Ouch, off-by-one error"
            swv_buffer[p_start:p_end] += \
                r['data'][r_start:r_end] * adc_to_pe[ch]

            right_r_i += 1

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
