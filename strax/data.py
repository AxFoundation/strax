"""Store pulses in memory-mapped numpy arrays with metadata
These slurp / dump very quickly  (I don't really understand why
they are so much faster than f.write(), np.save, etc)
"""
import os
import bz2
import json

import blosc
import zstd
import numpy as np
import numba

__all__ = 'record_dtype records_needed load save delete load_metadata save_metadata'.split()

# Testing on 300 MB data, after baseline subtraction
# blosc 1000 MB/sec, 3x reduction
# zstd: 300 MB/sec, 5x reduction (harder to install?)
# bz2: 23 MB/sec, 6.4x reduction
# Other algorithms (zlib, snappy, lzma) are all worse than this
# (worse = takes longer AND reduces less)
COMPRESSORS = dict(bz2=bz2, blosc=blosc, zstd=zstd)
COMPRESS_OPTIONS = dict(blosc=dict(shuffle=False))


def record_dtype(samples_per_record):
    return [
        ('channel', np.int16),
        # Waveform data. Note this is defined as a SIGNED integer, so we can
        # subtract baselines later.
        ('data', np.int16, samples_per_record),
        # Start time of the RECORD (not pulse)
        # After sorting on this, we are guaranteed to see each pulses' records
        # in proper order
        ('time', np.int64),
        # Total number of samples in pulse (not record)
        ('total_length', np.int16),
        # Which record is this?
        ('record_i', np.int16)
    ]


@numba.jit
def records_needed(pulse_length, samples_per_record):
    """Return records needed to store pulse_length samples"""
    return 1 + (pulse_length - 1) // samples_per_record


def load(filename, with_meta=False):
    metadata = load_metadata(filename)

    compressor = metadata['compressor']
    if compressor == 'none':
        data = np.load(filename + '.npy')
    else:
        with open(f'{filename}.{compressor}', mode='rb') as f:
            data = COMPRESSORS[compressor].decompress(f.read())
        data = np.frombuffer(data, dtype=eval(metadata['dtype']))

    if with_meta:
        return data, metadata
    return data


def _ext(compressor):
    if compressor == 'none':
        return 'npy'
    return compressor


def delete(filename):
    metadata = load_metadata(filename)
    comp = metadata['compressor']
    os.remove(filename + '.' + ('npy' if comp == 'none' else comp))
    os.remove(filename + '.json')


def save(filename, records, compressor='zstd', **metadata):
    assert isinstance(records, np.ndarray), "Please pass a numpy array"
    save_metadata(filename,
                  compressor=compressor,
                  dtype=records.dtype.descr.__repr__(),
                  **metadata)

    if compressor == 'none':
        np.save(filename + '.npy', records)
    else:
        d_comp = COMPRESSORS[compressor].compress(
            records,
            **COMPRESS_OPTIONS.get(compressor, dict()))
        with open(f'{filename}.{compressor}', 'wb') as f:
            f.write(d_comp)


def save_metadata(filename, **metadata):
    with open(filename + '.json', mode='w') as f:
        f.write(json.dumps(dict(**metadata)))


def load_metadata(filename):
    with open(filename + '.json', mode='r') as f:
        metadata = json.loads(f.read())
    return metadata