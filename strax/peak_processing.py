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
    # TODO: track area per channel too, so you can do posrec later
    # Meh, can wait until we have posrec and care
    if not len(hits):
        return
    offset = 0
    assert gap_threshold > left_extension + right_extension

    peak_start = hits[0]['time']
    peak_end = hits[0]['endtime']
    n_hits = 0

    for i, hit in enumerate(hits[1:]):
        gap = hit['time'] - peak_end
        if gap > gap_threshold or hit['time'] > peak_start + max_duration:
            # This hit no longer belongs to the same signal
            # store the old signal if it contains enough hits
            if n_hits >= min_hits:
                res = result_buffer[offset]
                res['time'] = peak_start - left_extension
                res['endtime'] = peak_end + right_extension
                res['n_hits'] = n_hits

                offset += 1
                if offset == len(result_buffer):
                    yield offset
                    offset = 0
            n_hits = 0
            peak_start = hit['time']
            peak_end = hit['endtime']

        else:
            # Hit continues the current signal
            peak_end = max(hit['endtime'], peak_end)
            n_hits += 1

    yield offset


@utils.growing_result(dtype=peak_dtype(0), chunk_size=int(1e4))
@numba.jit(nopython=True, nogil=True)
def find_large_peaks_roughly(
        result_buffer, records, to_pe,
        dt=10,
        gap_threshold=300,
        min_area=int(2e5),
        max_duration=1000):
    # This is a copy-pasted version of find_peaks, modified to work on records
    # and keep track of integral area instead of n_hits
    # TODO: find some way to avoid duplicated logic
    if not len(records):
        return
    offset = 0

    time = records['time']
    endtime = records['time'] + len(records[0]['data']) * dt

    # TODO: Hm, there's duplication here as well...
    peak_start = time[0]
    peak_end = endtime[0]
    area = records[0]['area'] * to_pe[records[0]['channel']]

    for r_i, r in enumerate(records[1:]):
        t = time[r_i]
        et = endtime[r_i]
        ar = r['area'] * to_pe[r['channel']]

        gap = t - peak_end

        if gap > gap_threshold or t > peak_start + max_duration:
            # This hit no longer belongs to the same signal
            # store the old signal if it contains enough hits
            if area >= min_area:
                res = result_buffer[offset]
                res['time'] = peak_start
                res['endtime'] = peak_end
                res['area'] = area

                offset += 1
                if offset == len(result_buffer):
                    yield offset
                    offset = 0
            peak_start = t
            peak_end = et
            area = ar

        else:
            # Hit continues the current signal
            peak_end = max(et, peak_end)
            area += ar

    yield offset


# TODO: remove hardcoded 10
@numba.jit(nopython=True, nogil=True)
def sum_waveform(peaks, records, adc_to_pe):
    """Compute sum waveforms for all peaks in peaks
    Will downsample sum waveforms if they do not fit in per-peak buffer
    """
    peak_lengths = (peaks['endtime'] - peaks['time']) // 10
    samples_per_record = len(records[0]['data'])
    time_per_record = samples_per_record * 10
    sum_wv_samples = len(peaks[0]['sum_waveform'])

    # Big buffer to hold even largest sum waveforms
    # Need a little more even for downsampling..
    swv_buffer = np.zeros(peak_lengths.max() * 2, dtype=np.float32)

    # Indices to a window of records
    left_r_i = 0
    right_r_i = 0

    for peak_i, p in enumerate(peaks):
        # Clear the relevant part of the swv buffer for use
        # (we clear a bit extra for use in downsampling)
        p_length = peak_lengths[peak_i]
        swv_buffer[:min(2 * p_length, len(swv_buffer))] = 0

        # Find first record that contributes to peak
        while records[left_r_i]['time'] + time_per_record <= p['time']:
            left_r_i += 1

        # Scan ahead over records that contribute
        right_r_i = left_r_i
        while True:
            r = records[right_r_i]
            ch = r['channel']

            s = int((p['time'] - r['time']) // 10)
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
        ds = p['downsample_factor'] = int(np.ceil(p_length / sum_wv_samples))
        if ds > 1:
            new_ns = int(np.ceil(p_length / ds)) * ds
            p['sum_waveform'][:new_ns // ds] = \
                swv_buffer[:new_ns].reshape(-1, ds).sum(axis=1)
        else:
            p['sum_waveform'][:p_length] = swv_buffer[:p_length]
