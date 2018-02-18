import numba
from strax import utils

from strax.data import peak_dtype


# TODO: remove hardcoded n_channels
@utils.growing_result(dtype=peak_dtype(260), chunk_size=int(1e4))
@numba.jit(nopython=True)
def find_peaks(result_buffer, hits, gap_threshold=150):
    if not len(hits):
        return
    offset = 0
    peak_start = hits[0]['time']
    peak_end = hits[0]['endtime']
    n_hits = 0
    for i, hit in enumerate(hits[1:]):
        gap = hit['time'] - peak_end
        if gap > gap_threshold:
            # This hit no longer belongs to the same signal
            # store the old signal if it contains > 1 hit
            if n_hits > 1:
                res = result_buffer[offset]
                res['time'] = peak_start
                res['endtime'] = peak_end

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
