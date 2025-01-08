import numpy as np
import numba

import strax

export, __all__ = strax.exporter()


@export
@numba.njit(cache=True, nogil=True)
def index_of_fraction(peaks, fractions_desired):
    """Return the (fractional) indices at which the peaks reach fractions_desired of their area.

    :param peaks: strax peak(let)s or other data-bearing dtype
    :param fractions_desired: array of floats between 0 and 1
    :return: (len(peaks), len(fractions_desired)) array of floats

    """
    results = np.zeros((len(peaks), len(fractions_desired)), dtype=np.float32)

    for p_i, p in enumerate(peaks):
        if p["area"] <= 0:
            continue  # TODO: These occur a lot. Investigate!
        compute_index_of_fraction(p, fractions_desired, results[p_i])
    return results


@export
@numba.njit(nogil=True, cache=True)
def compute_index_of_fraction(peak, fractions_desired, result):
    """Store the (fractional) indices at which peak reaches fractions_desired of their area in
    result.

    :param peak: single strax peak(let) or other data-bearing dtype
    :param fractions_desired: array of floats between 0 and 1
    :return: len(fractions_desired) array of floats

    """
    area_tot = peak["area"]
    fraction_seen = 0
    current_fraction_index = 0
    needed_fraction = fractions_desired[current_fraction_index]
    for i, x in enumerate(peak["data"][: peak["length"]]):
        # How much of the area is in this sample?
        fraction_this_sample = x / area_tot

        # Are we passing any desired fractions in this sample?
        while fraction_seen + fraction_this_sample >= needed_fraction:
            area_needed = area_tot * (needed_fraction - fraction_seen)
            if x != 0:
                result[current_fraction_index] = i + area_needed / x
            else:
                result[current_fraction_index] = i

            # Advance to the next fraction
            current_fraction_index += 1
            if current_fraction_index > len(fractions_desired) - 1:
                break
            needed_fraction = fractions_desired[current_fraction_index]

        if current_fraction_index > len(fractions_desired) - 1:
            break

        # Add this sample's area to the area seen
        fraction_seen += fraction_this_sample

    if needed_fraction == 1:
        # Sometimes floating-point errors prevent the full area
        # from being reached before the waveform ends
        result[-1] = peak["length"]


@export
def compute_widths(peaks):
    """Compute widths in ns at desired area fractions for peaks.

    :param peaks: single strax peak(let) or other data-bearing dtype

    """

    desired_widths = np.linspace(0, 1, len(peaks[0]["width"]))
    # 0% are width is 0 by definition, and it messes up the calculation below
    desired_widths = desired_widths[1:]

    # Which area fractions do we need times for?
    desired_fr = np.concatenate([0.5 - desired_widths / 2, 0.5 + desired_widths / 2])

    # We lose the 50% fraction with this operation, let's add it back
    desired_fr = strax.stable_sort(np.unique(np.append(desired_fr, [0.5])))

    fr_times = index_of_fraction(peaks, desired_fr)
    fr_times *= peaks["dt"].reshape(-1, 1)

    i = len(desired_fr) // 2
    median_time = fr_times[:, i]
    width = fr_times[:, i:] - fr_times[:, ::-1][:, i:]
    area_decile_from_midpoint = fr_times[:, ::2] - fr_times[:, i].reshape(-1, 1)
    return median_time, width, area_decile_from_midpoint


@numba.njit(cache=True, nogil=True)
def compute_center_time(peaks):
    """Compute the center time of the peaks.

    :param peaks: single strax peak(let) or other data-bearing dtype

    """
    center_time = np.zeros(len(peaks), dtype=np.int64)
    for p_i, p in enumerate(peaks):
        data = p["data"][: p["length"]]
        if p["area"] <= 0:
            # Negative or zero-area peaks have centertime at startime
            center_time[p_i] = p["time"]
        else:
            t = np.average(np.arange(p["length"]), weights=data)
            center_time[p_i] = (t + 1 / 2) * p["dt"]
            center_time[p_i] += p["time"]  # converting from float to int, implicit floor
    return center_time


@export
@numba.njit(cache=True, nogil=True)
def compute_area_fraction_top(peaks, n_top_channels):
    """Compute the area fraction top for peaks."""
    area_fraction_top = np.zeros(len(peaks), dtype=np.float32)
    for peak_i in range(len(peaks)):
        p = peaks[peak_i]
        area_top = p["area_per_channel"][:n_top_channels].sum()
        # Non-positive-area peaks get NaN AFT
        if p["area"] > 0:
            area_fraction_top[peak_i] = area_top / p["area"]
        else:
            area_fraction_top[peak_i] = np.nan
    return area_fraction_top


@export
def compute_properties(peaks, n_top_channels=0, select_peaks_indices=None):
    """Compute properties: median_time, width, area_decile_from_midpoint,
    center_time, and area_fraction_top for peaks.

    :param peaks: single strax peak(let) or other data-bearing dtype
    :param select_peaks_indices: array of integers informing which peaks to compute default to None
        in which case compute for all peaks

    """
    if not len(peaks) or (select_peaks_indices is not None and not len(select_peaks_indices)):
        return

    if select_peaks_indices is None:
        select_peaks_indices = slice(None)

    median_time, width, area_decile_from_midpoint = compute_widths(peaks[select_peaks_indices])
    peaks["median_time"][select_peaks_indices] = median_time
    peaks["width"][select_peaks_indices] = width
    peaks["area_decile_from_midpoint"][select_peaks_indices] = area_decile_from_midpoint

    center_time = compute_center_time(peaks[select_peaks_indices])
    peaks["center_time"][select_peaks_indices] = center_time

    if n_top_channels > 0:
        area_fraction_top = compute_area_fraction_top(peaks[select_peaks_indices], n_top_channels)
        peaks["area_fraction_top"][select_peaks_indices] = area_fraction_top
