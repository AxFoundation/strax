import os
import shutil

import numpy as np

from .data import save_records


def reader_split(records, output_dir, n_readers=8):
    """Split records over n_readers files by channels
    then output them in output_dir (deleting dir if exists)
    """
    n_channels = records['channel'].max() + 1
    channels_per_reader = np.ceil(n_channels / n_readers)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for reader_i in range(n_readers):
        first_channel = reader_i * channels_per_reader
        reader_data = records[
            (records['channel'] >= first_channel) &
            (records['channel'] < first_channel + channels_per_reader)]
        save_records(f'{output_dir}/reader_{reader_i}.bin', reader_data)
