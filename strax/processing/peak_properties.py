import numpy as np
import numba

__all__ = 'compute_widths'.split()


def index_of_fraction(peaks, fractions_desired):
    # nopython does not allow this dynamic allocation:
    results = np.zeros((len(peaks), len(fractions_desired)), dtype=np.float)
    _index_of_fraction(peaks, fractions_desired, results)
    return results


@numba.jit(nopython=True, nogil=True, cache=True)
def _index_of_fraction(peaks, fractions_desired, results):
    for p_i, p in enumerate(peaks):
        area_tot = p['area']
        if area_tot <= 0:
            continue  # TODO: These occur a lot. Investigate!
        r = results[p_i]

        fraction_seen = 0
        current_fraction_index = 0
        needed_fraction = fractions_desired[current_fraction_index]
        for i, x in enumerate(p['data'][:p['length']]):
            # How much of the area is in this sample?
            fraction_this_sample = x / area_tot

            # Are we passing any desired fractions in this sample?
            while fraction_seen + fraction_this_sample >= needed_fraction:

                area_needed = area_tot * (needed_fraction - fraction_seen)
                if x != 0:
                    r[current_fraction_index] = i + area_needed / x
                else:
                    r[current_fraction_index] = i

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
            r[-1] = p['length']

    return results


def compute_widths(peaks):
    """Compute widths in ns at desired area fractions for peaks
    returns (n_peaks, n_widths) array
    """
    if not len(peaks):
        return

    desired_widths = np.linspace(0, 1, len(peaks[0]['width']))
    # 0% are width is 0 by definition, and it messes up the calculation below
    desired_widths = desired_widths[1:]

    # Which area fractions do we need times for?
    desired_fr = np.concatenate([0.5 - desired_widths / 2,
                                 0.5 + desired_widths / 2])
    desired_fr = np.sort(np.unique(desired_fr))

    fr_times = index_of_fraction(peaks, desired_fr)
    fr_times *= peaks['dt'].reshape(-1, 1)

    i = len(desired_fr) // 2
    peaks['width'][:, 1:] = fr_times[:, i:] - fr_times[:, :i]
