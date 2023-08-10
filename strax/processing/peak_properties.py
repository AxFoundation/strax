import numpy as np
import numba

import strax
export, __all__ = strax.exporter()


@export
@numba.njit(cache=True, nogil=True)
def index_of_fraction(peaks, fractions_desired):
    """Return the (fractional) indices at which the peaks reach
    fractions_desired of their area
    :param peaks: strax peak(let)s or other data-bearing dtype
    :param fractions_desired: array of floats between 0 and 1
    :returns: (len(peaks), len(fractions_desired)) array of floats
    """
    results = np.zeros((len(peaks), len(fractions_desired)), dtype=np.float32)

    for p_i, p in enumerate(peaks):
        if p['area'] <= 0:
            continue  # TODO: These occur a lot. Investigate!
        compute_index_of_fraction(p, fractions_desired, results[p_i])
    return results


@export
@numba.jit(nopython=True, nogil=True, cache=True)
def compute_index_of_fraction(peak, fractions_desired, result):
    """Store the (fractional) indices at which peak reaches
    fractions_desired of their area in result
    :param peak: single strax peak(let) or other data-bearing dtype
    :param fractions_desired: array of floats between 0 and 1
    :returns: len(fractions_desired) array of floats
    """
    area_tot = peak['area']
    fraction_seen = 0
    current_fraction_index = 0
    needed_fraction = fractions_desired[current_fraction_index]
    for i, x in enumerate(peak['data'][:peak['length']]):
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
        result[-1] = peak['length']


@export
def compute_widths(peaks, select_peaks_indices=None):
    """Compute widths in ns at desired area fractions for peaks
    :param peaks: single strax peak(let) or other data-bearing dtype
    :param select_peaks_indices: array of integers informing which peaks to compute
        default to None in which case compute for all peaks
    """
    if not len(peaks):
        return
    if select_peaks_indices is None:
        select_peaks_indices = np.arange(len(peaks))
    if isinstance(select_peaks_indices, list):
        select_peaks_indices = np.array(select_peaks_indices, int)
    if not len(select_peaks_indices):
        return

    desired_widths = np.linspace(0, 1, len(peaks[0]['width']))
    # 0% are width is 0 by definition, and it messes up the calculation below
    desired_widths = desired_widths[1:]

    # Which area fractions do we need times for?
    desired_fr = np.concatenate([0.5 - desired_widths / 2,
                                 0.5 + desired_widths / 2])

    # We lose the 50% fraction with this operation, let's add it back
    desired_fr = np.sort(np.unique(np.append(desired_fr, [0.5])))

    fr_times = index_of_fraction(peaks[select_peaks_indices], desired_fr)
    fr_times *= peaks['dt'][select_peaks_indices].reshape(-1, 1)

    i = len(desired_fr) // 2
    peaks['width'][select_peaks_indices] = fr_times[:, i:] - fr_times[:, ::-1][:, i:]
    peaks['area_decile_from_midpoint'][select_peaks_indices] = fr_times[:, ::2] - fr_times[:, i].reshape(-1,1)

@export
@numba.jit(nopython=True, cache=True)
def compute_wf_attributes(data, sample_length, n_samples: int, downsample_wf=False):
    """
    Compute waveform attribures
    Quantiles: represent the amount of time elapsed for
    a given fraction of the total waveform area to be observed in n_samples
    i.e. n_samples = 10, then quantiles are equivalent deciles 
    Waveforms: downsampled waveform to n_samples
    :param data: waveform e.g. peaks or peaklets
    :param n_samples: compute quantiles for a given number of samples 
    :return: waveforms and quantiles of size n_samples
    """    
    assert data.shape[0] == len(sample_length), "ararys must have same size"

    num_samples = data.shape[1]

    waveforms = np.zeros((len(data), n_samples), dtype=np.float64)
    quantiles = np.zeros((len(data), n_samples), dtype=np.float64)

    # Cannot compute with with more samples than actual waveform sample
    assert num_samples > n_samples, "cannot compute with more samples than the actual waveform"


    step_size = int(num_samples / n_samples)
    steps = np.arange(0, num_samples + 1, step_size)
    inter_points = np.linspace(0., 1. - (1. / n_samples), n_samples)
    cumsum_steps = np.zeros(n_samples + 1, dtype=np.float64)
    frac_of_cumsum = np.zeros(num_samples + 1)
    sample_number_div_dt = np.arange(0, num_samples + 1, 1)
    for i, (samples, dt) in enumerate(zip(data, sample_length)):
        if np.sum(samples) == 0: 
            continue
        # reset buffers
        frac_of_cumsum[:] = 0
        cumsum_steps[:] = 0
        frac_of_cumsum[1:] = np.cumsum(samples)
        frac_of_cumsum[1:] = frac_of_cumsum[1:] / frac_of_cumsum[-1]
        cumsum_steps[:-1] = np.interp(inter_points, frac_of_cumsum, sample_number_div_dt * dt)
        cumsum_steps[-1] = sample_number_div_dt[-1] * dt
        quantiles[i] = cumsum_steps[1:] - cumsum_steps[:-1]
   
        if downsample_wf:
            
            for j in range(n_samples):
                waveforms[i][j] = np.sum(samples[steps[j]:steps[j + 1]])
            waveforms[i] /= (step_size * dt)
    return quantiles, waveforms
