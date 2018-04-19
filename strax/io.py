from ast import literal_eval
from functools import partial
import os
import bz2
import json

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
def load(filename, compressor=None, dtype=None, return_meta=False):
    """Read and return data from filename

    :param compressor: compressor to use for decompressing. If not passed,
    will try to load it from json metadata file.
    :param return_meta: if True, return (data, metadata) tuple
    """
    # Remove extension from filename (if present)
    # Let's hope nobody puts extra dots in the filename...
    filename = os.path.splitext(filename)[0]

    if compressor is None or dtype is None:
        metadata = load_metadata(filename)
        compressor = metadata['compressor']
        dtype = literal_eval(metadata['dtype'])
    if compressor == 'none':
        data = np.load(filename)
    else:
        with open(filename, mode='rb') as f:
            data = COMPRESSORS[compressor]['decompress'](f.read())
        # frombuffer is much faster, but results in some readonly fields
        data = np.frombuffer(data, dtype=dtype)

    if return_meta:
        return data, metadata
    return data


@export
def save(filename, data, compressor='zstd', save_meta=True, **metadata):
    """Save data to filename, return filesize in bytes
    :param compressor: compressor to use
    :param save_meta: If False, just save to filename.
    If True, save data to filename and metadata to filename + .json.
    Metadata includes dtype, compressor, and any
    additional kwargs passed to save.
    """
    assert isinstance(data, np.ndarray), "Please pass a numpy array"
    if save_meta:
        save_metadata(filename,
                      compressor=compressor,
                      dtype=data.dtype.descr.__repr__(),
                      **metadata)
    if compressor == 'none':
        np.save(filename, data)
    else:
        d_comp = COMPRESSORS[compressor]['compress'](data)
        with open(filename, 'wb') as f:
            f.write(d_comp)

    return os.path.getsize(filename)


@export
def save_metadata(filename, **metadata):
    with open(filename + '.json', mode='w') as f:
        f.write(json.dumps(dict(**metadata)))


@export
def load_metadata(filename):
    with open(filename + '.json', mode='r') as f:
        metadata = json.loads(f.read())
    return metadata
