import numba
import numpy as np

import strax
from strax.processing.general import _fully_contained_in, _fully_contained_in_sanity

export, __all__ = strax.exporter()


@export
def merge_peaks(peaks, start_merge_at, end_merge_at, max_buffer=int(1e5)):
    """Merge specified peaks with their neighbors, return merged peaks.

    :param peaks: Record array of strax peak dtype.
    :param start_merge_at: Indices to start merge at
    :param end_merge_at: EXCLUSIVE indices to end merge at
    :param max_buffer: Maximum number of samples in the sum_waveforms and other waveforms of the
        resulting peaks (after merging). Peaks must be constructed based on the properties of
        constituent peaks, it being too time-consuming to revert to records/hits.

    """
    assert len(start_merge_at) == len(end_merge_at)
    new_peaks = np.zeros(len(start_merge_at), dtype=peaks.dtype)

    # Do the merging. Could numbafy this to optimize, probably...
    buffer = np.zeros(max_buffer, dtype=np.float32)
    buffer_top = np.zeros(max_buffer, dtype=np.float32)

    for new_i, new_p in enumerate(new_peaks):
        old_peaks = peaks[start_merge_at[new_i] : end_merge_at[new_i]]
        common_dt = np.gcd.reduce(old_peaks["dt"])
        first_peak, last_peak = old_peaks[0], old_peaks[-1]
        new_p["channel"] = first_peak["channel"]

        # The new endtime must be at or before the last peak endtime
        # to avoid possibly overlapping peaks
        new_p["time"] = first_peak["time"]
        new_p["dt"] = common_dt
        new_p["length"] = (strax.endtime(last_peak) - new_p["time"]) // common_dt

        # re-zero relevant part of buffers (overkill? not sure if
        # this saves much time)
        buffer[
            : min(
                int(
                    (
                        last_peak["time"]
                        - first_peak["time"]
                        + (last_peak["length"] * old_peaks["dt"].max())
                    )
                    / common_dt
                ),
                len(buffer),
            )
        ] = 0
        buffer_top[
            : min(
                int(
                    (
                        last_peak["time"]
                        - first_peak["time"]
                        + (last_peak["length"] * old_peaks["dt"].max())
                    )
                    / common_dt
                ),
                len(buffer_top),
            )
        ] = 0

        for p in old_peaks:
            # Upsample the sum and top/bottom array waveforms into their buffers
            upsample = p["dt"] // common_dt
            n_after = p["length"] * upsample
            i0 = (p["time"] - new_p["time"]) // common_dt
            buffer[i0 : i0 + n_after] = np.repeat(p["data"][: p["length"]], upsample) / upsample
            buffer_top[i0 : i0 + n_after] = (
                np.repeat(p["data_top"][: p["length"]], upsample) / upsample
            )

            # Handle the other peak attributes
            new_p["area"] += p["area"]
            new_p["area_per_channel"] += p["area_per_channel"]
            new_p["n_hits"] += p["n_hits"]
            new_p["saturated_channel"][p["saturated_channel"] == 1] = 1

        # Downsample the buffers into new_p['data'], new_p['data_top'],
        # and new_p['data_bot']
        strax.store_downsampled_waveform(new_p, buffer, True, buffer_top)

        new_p["n_saturated_channels"] = new_p["saturated_channel"].sum()

        # Use tight_coincidence of the peak with the highest amplitude
        i_max_subpeak = old_peaks["data"].max(axis=1).argmax()
        new_p["tight_coincidence"] = old_peaks["tight_coincidence"][i_max_subpeak]

        # Too lazy to compute these
        for p in "max_gap max_diff min_diff".split():
            new_p[p] = -1
        new_p["max_goodness_of_split"] = np.nan

        # If the endtime was in the peaks we have to recompute it here
        # because otherwise it will stay set to zero due to the buffer
        if "endtime" in new_p.dtype.names:
            new_p["endtime"] = strax.endtime(last_peak)
    return new_peaks


@export
def replace_merged(orig, merge):
    """Return sorted array of 'merge' and members of 'orig' that do not touch any of merge.

    :param orig: Array of interval-like objects (e.g. peaks)
    :param merge: Array of interval-like objects (e.g. peaks)

    """
    if not len(merge):
        return orig

    skip_windows = strax.touching_windows(orig, merge)
    skip_n = np.diff(skip_windows, axis=1).sum()
    result = np.zeros(len(orig) - skip_n + len(merge), dtype=orig.dtype)
    _replace_merged(result, orig, merge, skip_windows)
    return result


@numba.njit(cache=True, nogil=True)
def _replace_merged(result, orig, merge, skip_windows):
    result_i = window_i = 0
    skip_start, skip_end = skip_windows[0]
    n_orig = len(orig)

    n_skipped = 0

    for orig_i in range(n_orig):
        if orig_i == skip_end:
            result[result_i] = merge[window_i]
            result_i += 1

            window_i += 1
            if window_i == len(skip_windows):
                skip_start = skip_end = n_orig + 100
            else:
                skip_start, skip_end = skip_windows[window_i]

        if orig_i >= skip_start:
            n_skipped += 1
            continue

        result[result_i] = orig[orig_i]
        result_i += 1

    if skip_end == n_orig:
        # Still have to insert the last merged S2
        # since orig_i == skip_end is never met
        assert result_i == len(result) - 1
        assert window_i == len(merge) - 1
        result[result_i] = merge[window_i]
        result_i += 1
        window_i += 1

    assert result_i == len(result)
    assert window_i == len(skip_windows)


@export
def add_lone_hits(peaks, lone_hits, to_pe, n_top_channels=0):
    """Function which adds information from lone hits to peaks if lone hit is inside a peak (e.g.
    after merging.). Modifies peak area and data inplace.

    :param peaks: Numpy array of peaks
    :param lone_hits: Numpy array of lone_hits
    :param to_pe: Gain values to convert lone hit area into PE.
    :param n_top_channels: Number of top array channels.

    """
    _fully_contained_in_sanity(lone_hits, peaks)
    _add_lone_hits(peaks, lone_hits, to_pe, n_top_channels=0)


@numba.njit(cache=True, nogil=True)
def _add_lone_hits(peaks, lone_hits, to_pe, n_top_channels=0):
    """The core function of add_lone_hits."""
    fully_contained_index = _fully_contained_in(lone_hits, peaks)

    for fc_i, lh_i in zip(fully_contained_index, lone_hits):
        if fc_i == -1:
            continue
        p = peaks[fc_i]
        lh_area = lh_i["area"] * to_pe[lh_i["channel"]]
        p["area"] += lh_area
        p["area_per_channel"][lh_i["channel"]] += lh_area

        # Add lone hit as delta pulse to waveform:
        index = (lh_i["time"] - p["time"]) // p["dt"]
        if index < 0 or index > len(p["data"]):
            raise ValueError("Hit outside of full containment!")
        p["data"][index] += lh_area

        if n_top_channels > 0:
            if lh_i["channel"] < n_top_channels:
                p["data_top"][index] += lh_area
