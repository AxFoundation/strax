import os
import pandas as pd
import glob
import shutil

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    # Mock out tqdm. Not necessary to add it as dependency
    def tqdm(x, *args, **kwargs):
        return x

import strax


class Saver:
    def __init__(self, dirpath, compressor='blosc'):
        self.compressor = compressor
        self.buffer = []
        self.dir = dirpath
        # Clean the dir if it already exists.
        # Hmm, a little dangerous. What if someone tries '.'?
        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        os.makedirs(self.dir)

        self.chunks_saved = 0

    def feed(self, x):
        self.buffer.append(x)
        if self.ready_to_write():
            self.write()

    def write(self):
        fn = os.path.join(self.dir, '%06d' % self.chunks_saved)
        r = np.concatenate(self.buffer)

        # TODO: is this right? is this the right place?
        metadata = dict()
        if 'time' in r[0]:
            metadata['first_time'] = int(r[0]['time'])

        strax.save(fn, r,
                   compressor=self.compressor,
                   **metadata)

        self.buffer = []
        self.chunks_saved += 1

    def flush(self):
        if len(self.buffer):
            self.write()
        self.chunks_saved += 1

    def ready_to_write(self):
        return True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.flush()


class GroupNSaver(Saver):

    def __init__(self, *args, group_chunks=5, **kwargs):
        self.group_chunks = group_chunks
        super().__init__(*args, **kwargs)

    def ready_to_write(self):
        return len(self.buffer) > self.group_chunks


class ThresholdSizeSaver(Saver):
    """
    Note the threshold size is on the *raw* buffered data, but saving is
    compressed (blosc) by default, so you'll get much smaller files than
    you expect.
    """

    def __init__(self, *args, threshold_bytes=100 * int(1e6), **kwargs):
        self.threshold_bytes = threshold_bytes
        super().__init__(*args, **kwargs)

    def ready_to_write(self):
        buffer_bytes = sum(x.nbytes for x in self.buffer)
        return buffer_bytes >= self.threshold_bytes


def chunk_files(dn):
    """Return sorted list of strax chunk file name in directory dn
    without file extension (so ready from strax.load)
    """
    if not os.path.exists(dn):
        raise FileNotFoundError(dn)
    files = sorted(glob.glob(os.path.join(dn,'*.json')))
    return [os.path.splitext(f)[0] for f in files]


def read_chunks(dn, desc=None, **kwargs):
    """Iteratively read strax chunk files in directory path dn"""
    it = chunk_files(dn)
    if desc is not None:
        # Add progress bar
        it = tqdm(it, desc=desc)
    for f in it:
        yield strax.load(f)


def get_chunk_starts(dn):
    """Return numpy array of chunk start times in dn
    Cache this in chunk_info.npy
    """
    info_fn = os.path.join(dn, 'chunk_info.npy')
    if os.path.exists(info_fn):
        return np.load(info_fn)

    chunk_starts = np.array([strax.load_metadata(fn)['first_time']
                             for fn in chunk_files(dn)],
                            dtype=np.int64)
    np.save(info_fn, chunk_starts)
    return chunk_starts


def slurp(dn):
    """Return concatenated array with data of all chunks in dn"""
    return np.concatenate(list(read_chunks(dn)))


def slurp_df(dn):
    """Return concatenated dataframe with data of all chunks in dn"""
    return pd.DataFrame.from_records(slurp(dn))
