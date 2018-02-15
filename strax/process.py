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
def baseline(records, baseline_samples=40):
    """Subtract pulses from int(baseline), store baseline in baseline field
    :param baseline_samples: number of samples at start of pulse to average
    Assumes records are sorted in time (or at least by channel, then time)
    """
    if not len(records):
        return
    samples_per_record = len(records[0]['data'])

    # Array for looking up last baseline seen in channel
    # We only care about the channels in this set of records; a single .max()
    # is worth avoiding the hassle of passing n_channels around
    last_bl_in = np.zeros(records['channel'].max() + 1, dtype=np.int16)

    for d_i, d in enumerate(records):

        # Compute the baseline if we're the first record of the pulse,
        # otherwise take the last baseline we've seen in the channel
        if d.record_i == 0:
            bl = last_bl_in[d.channel] = d.data[:baseline_samples].mean()
        else:
            bl = last_bl_in[d.channel]

        # Subtract baseline from all data samples in the record
        # (any additional zeros are already zero)
        last = min(samples_per_record,
                   d.total_length - d.record_i * samples_per_record)
        d.data[:last] = int(bl) - d.data[:last]
        d.baseline = bl
