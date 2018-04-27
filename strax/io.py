"""Read/write numpy arrays to/from compressed files
"""
from functools import partial
import os
import bz2

import numpy as np
import blosc
import zstd

import strax
export, __all__ = strax.exporter()

blosc.set_releasegil(True)


COMPRESSORS = dict(
    bz2=dict(
        compress=bz2.compress,
        decompress=bz2.decompress),
    zstd=dict(
        compress=zstd.compress,
        decompress=zstd.decompress),
    blosc=dict(
        compress=partial(blosc.compress, shuffle=False),
        decompress=blosc.decompress),
)


@export
def load_file(filename, compressor=None, dtype=None):
    """Read and return data from filename

    :param compressor: compressor to use for decompressing. If not passed,
    will try to load it from json metadata file.
    """
    if compressor == 'none':
        data = np.load(filename)
    else:
        with open(filename, mode='rb') as f:
            data = COMPRESSORS[compressor]['decompress'](f.read())
        data = np.frombuffer(data, dtype=dtype)
    return data


@export
def save_file(filename, data, compressor='zstd'):
    """Save data to filename, return filesize in bytes
    :param compressor: compressor to use
    """
    assert isinstance(data, np.ndarray), "Please pass a numpy array"
    if compressor == 'none':
        np.save(filename, data)
    else:
        d_comp = COMPRESSORS[compressor]['compress'](data)
        with open(filename, 'wb') as f:
            f.write(d_comp)

    return os.path.getsize(filename)
