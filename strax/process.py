import glob

import numpy as np
import numba

from .data import save, load


def process(input_dir, output_filename):
    records = load_from_readers(input_dir)
    _ = baseline(records)
    save(output_filename, records)


def load_from_readers(input_dir):
    """Return concatenated & sorted records from multiple reader data files"""
    records = [load(f)
               for f in glob.glob(f'{input_dir}/reader_?.bin')]
    records = np.concatenate(records)
    records = sort_by_time(records)
    return records


# ~7x faster than np.sort(records, order='time'). Try it.
@numba.jit(nopython=True)
def sort_by_time(records):
    time = records['time'].copy()
    sort_i = np.argsort(time)
    return records[sort_i]


@numba.jit(nopython=True)
def baseline(records, n_before=48, n_after=30):
    """Determine baseline from n_after samples and subtract it from pulses.
    To be more precise:
      - Determine mean M of the first n_before samples in pulse
      - waveform = int(M) - waveform
      - Zero n_before and n_after samples at start and end of pulse, resp.
      - Zero any junk padding data
      - Returns array of M (float32 array) of baseline corr. to each fragment
    """
    # TODO: records.data.shape[1] gives a numba error (file issue?)
    if not len(records):
        return
    samples_per_record = len(records[0]['data'])

    # Array for looking up last baseline seen in channel
    # We only care about the channels in this set of records; a single .max()
    # is worth avoiding the hassle of passing n_channels around
    last_bl_in_channel = np.zeros(records['channel'].max() + 1,
                                  dtype=np.int16)

    all_baselines = np.zeros(len(records), dtype=np.float32)

    for d_i, d in enumerate(records):

        if d.record_i == 0:
            # This is the first record of the pulse: determine baseline
            all_baselines[d_i] = bl = last_bl_in_channel[d.channel] = \
                d.data[:n_before].mean()

        else:
            # Secondary record: use baseline from first record
            all_baselines[d_i] = bl = last_bl_in_channel[d.channel]

        # Do the baseline subtraction
        d.data[:] = int(bl) - d.data[:]

        # Zero leading baseline
        if d.record_i == 0:
            d.data[:n_before] = 0

        # Zero trailing baseline & junk samples
        in_pulse_clear_from = d.total_length - n_after
        in_record_clear_from = max(0, in_pulse_clear_from
                                      - d.record_i * samples_per_record)
        if in_record_clear_from < samples_per_record:
            d.data[in_record_clear_from:] = 0

    return all_baselines
