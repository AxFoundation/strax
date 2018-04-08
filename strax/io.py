from functools import partial
import os
import bz2
import json

import numpy as np
import blosc
import zstd

import strax
export, __all__ = strax.exporter()


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
def load(filename, with_meta=False):
    # Remove extension from filename (if present)
    # Let's hope nobody puts extra dots in the filename...
    filename = os.path.splitext(filename)[0]

    metadata = load_metadata(filename)
    compressor = metadata['compressor']
    if compressor == 'none':
        data = np.load(filename + '.npy')
    else:
        with open(_fn(filename, compressor), mode='rb') as f:
            data = COMPRESSORS[compressor]['decompress'](f.read())
        # frombuffer is much faster, but results in some readonly fields
        data = np.fromstring(data, dtype=eval(metadata['dtype']))

    if with_meta:
        return data, metadata
    return data


@export
def delete(filename):
    metadata = load_metadata(filename)
    os.remove(_fn(filename, metadata['compressor']))
    os.remove(filename + '.json')


@export
def save(filename, records, compressor='zstd', **metadata):
    assert isinstance(records, np.ndarray), "Please pass a numpy array"
    save_metadata(filename,
                  compressor=compressor,
                  dtype=records.dtype.descr.__repr__(),
                  **metadata)

    if compressor == 'none':
        np.save(filename + '.npy', records)
    else:
        d_comp = COMPRESSORS[compressor]['compress'](records)
        with open(_fn(filename, compressor), 'wb') as f:
            f.write(d_comp)


@export
def save_metadata(filename, **metadata):
    with open(filename + '.json', mode='w') as f:
        f.write(json.dumps(dict(**metadata)))


@export
def load_metadata(filename):
    with open(filename + '.json', mode='r') as f:
        metadata = json.loads(f.read())
    return metadata


@export
def _fn(filename, compressor):
    """Get filename (with extension) of data file"""
    if compressor == 'none':
        return filename + '.npy'
    return filename + '.' + COMPRESSORS[compressor].get('extension',
                                                        compressor)
