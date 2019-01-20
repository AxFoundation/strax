"""Read/write numpy arrays to/from compressed files or file-like objects
"""
from functools import partial
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
def load_file(f, compressor, dtype):
    """Read and return data from file

    :param f: file name or handle to read from
    :param compressor: compressor to use for decompressing. If not passed,
        will try to load it from json metadata file.
    :param dtype: numpy dtype of data to load
    """
    if isinstance(f, str):
        with open(f, mode='rb') as f:
            return _load_file(f, compressor, dtype)
    else:
        return _load_file(f, compressor, dtype)


def _load_file(f, compressor, dtype):
    try:
        data = f.read()
        if not len(data):
            return np.zeros(0, dtype=dtype)

        data = COMPRESSORS[compressor]['decompress'](data)
        return np.frombuffer(data, dtype=dtype)

    except Exception:
        raise strax.DataCorrupted(
            f"Fatal Error while reading file {f}: "
            + strax.utils.formatted_exception())


@export
def save_file(f, data, compressor='zstd'):
    """Save data to file and return number of bytes written

    :param f: file name or handle to save to
    :param data: data (numpy array) to save
    :param compressor: compressor to use
    """
    if isinstance(f, str):
        with open(f, mode='wb') as f:
            return _save_file(f, data, compressor)
    else:
        return _save_file(f, data, compressor)


def _save_file(f, data, compressor='zstd'):
    assert isinstance(data, np.ndarray), "Please pass a numpy array"
    d_comp = COMPRESSORS[compressor]['compress'](data)
    f.write(d_comp)
    return len(d_comp)
