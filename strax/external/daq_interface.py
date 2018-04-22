import glob
import shutil

import numpy as np

import strax

compressor = 'zstd'


def reader_split(records, n_readers=8):
    """Yields records split over n_readers
    """
    n_channels = records['channel'].max() + 1
    channels_per_reader = np.ceil(n_channels / n_readers)

    for reader_i in range(n_readers):
        first_channel = reader_i * channels_per_reader
        yield records[
            (records['channel'] >= first_channel) &
            (records['channel'] < first_channel + channels_per_reader)]


def load_from_readers(input_dir, erase=True):
    """Return concatenated & sorted records from multiple reader data files"""
    records = [strax.load(f,
                          compressor=compressor,
                          dtype=strax.record_dtype())
               for f in glob.glob(f'{input_dir}/reader_*')]
    records = np.concatenate(records)
    records = strax.sort_by_time(records)
    if erase:
        shutil.rmtree(input_dir)
    return records
