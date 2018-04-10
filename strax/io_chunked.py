import os
import glob
import shutil


import numpy as np
from tqdm import tqdm

import strax


def save_to_dir(source, dirname, compressor='blosc'):
    """Iterate over source and save results to dirname"""
    # Clean the dir if it already exists.
    # Hmm, a little dangerous. What if someone tries '.'?
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)

    for n_saved, x in enumerate(source):
        fn = os.path.join(dirname, '%06d' % n_saved)

        # TODO: is this right? is this the right place?
        metadata = dict()
        if 'time' in x[0].dtype.names:
            metadata['first_time'] = int(x[0]['time'])

        strax.save(fn, x, compressor=compressor, **metadata)


def chunk_files(dn):
    """Return sorted list of strax chunk file name in directory dn
    without file extension (so ready from strax.load)
    """
    if not os.path.exists(dn):
        raise FileNotFoundError(dn)
    files = sorted(glob.glob(os.path.join(dn, '*.json')))
    return [os.path.splitext(f)[0] for f in files]


def read_chunks(dn, desc=None, **kwargs):
    """Iteratively read strax chunk files in directory path dn"""
    it = chunk_files(dn)
    if not len(it):
        print(f"No strax files in {dn}?")
    if desc is not None:
        # Add progress bar
        it = tqdm(it, desc=desc)
    for f in it:
        yield strax.load(f, **kwargs)


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
