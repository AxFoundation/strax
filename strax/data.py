import os
import bz2
import json

import blosc
import zstd
import numpy as np
import numba

__all__ = ('record_dtype records_needed load save delete '
           'load_metadata save_metadata').split()

COMPRESSORS = dict(bz2=bz2, blosc=blosc, zstd=zstd)
COMPRESS_OPTIONS = dict(blosc=dict(shuffle=False))


def record_dtype(samples_per_record):
    return [
        ('channel', np.int16),
        # Waveform data. Note this is defined as a SIGNED integer, so we can
        # still represent negative values after subtracting baselines
        ('data', np.int16, samples_per_record),
        # Start time of the RECORD (not the pulse!)
        # After sorting on this, we are guaranteed to see each pulses' records
        # in proper order
        ('time', np.int64),
        # Total number of samples in pulse (not the record!)
        ('total_length', np.int16),
        # Record index in pulse (0=first record, etc)
        ('record_i', np.int16),
        # Level of data reduction applied (strax.Reduction enum)
        ('reduction_level', np.uint8),
        # Original baseline in ADC counts.
        ('baseline', np.float32),
    ]


@numba.jit
def records_needed(pulse_length, samples_per_record):
    """Return records needed to store pulse_length samples"""
    return 1 + (pulse_length - 1) // samples_per_record


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
            data = COMPRESSORS[compressor].decompress(f.read())
        # If I do frombuffer here, I get a readonly thing on windows
        # Check on linux? Big performance difference?
        data = np.fromstring(data, dtype=eval(metadata['dtype']))

    if with_meta:
        return data, metadata
    return data


def delete(filename):
    metadata = load_metadata(filename)
    os.remove(_fn(filename, metadata['compressor']))
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
        with open(_fn(filename, compressor), 'wb') as f:
            f.write(d_comp)


def save_metadata(filename, **metadata):
    with open(filename + '.json', mode='w') as f:
        f.write(json.dumps(dict(**metadata)))


def load_metadata(filename):
    with open(filename + '.json', mode='r') as f:
        metadata = json.loads(f.read())
    return metadata


def _fn(filename, compressor):
    """Get filename (with extension) of data file"""
    if compressor == 'none':
        return filename + '.npy'
    return filename + '.' + compressor