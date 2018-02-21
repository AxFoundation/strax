from functools import partial
import os
import bz2
import json

import blosc
import zstd
import numpy as np
import numba

__all__ = ('record_dtype hit_dtype peak_dtype '
           'records_needed load save delete '
           'load_metadata save_metadata').split()

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

# Base dtype for interval-like objects (pulse, peak, hit)
interval_dtype = [
    # Channel number to which this interval applies
    ('channel', np.int16),
    # Sample time resolution in nanoseconds
    ('dt', np.int16),
    # Start time of the interval (ns since unix epoch)
    ('time', np.int64),
    # Length of the interval in samples
    ('length', np.int16),
]


def record_dtype(samples_per_record):
    """Data type for a waveform record"""
    return interval_dtype + [
        # Total samples in the PULSE to which this record belongs
        # (excluding zero-padding in the last record)
        ('pulse_length', np.int16),
        # Position of this record in the pulse
        ('record_i', np.int16),
        # Baseline in ADC counts
        # If this field is nonzero, the record has been int-baselined
        # That is, data = int(baseline) - original_data
        ('baseline', np.float32),
        # Integral in ADC x samples
        ('area', np.int32),
        # Level of data reduction applied (strax.Reduction enum)
        ('reduction_level', np.uint8),
        # Waveform data. Note this is defined as a SIGNED integer, so we can
        # still represent negative values after subtracting baselines
        ('data', np.int16, samples_per_record),
    ]


# Data type for a 'hit': a sub-range of a record
hit_dtype = interval_dtype + [
    # Start sample of hit in record
    ('left', np.int16),
    # End sample of hit in record - exclusive bound! (like python ranges)
    ('right', np.int16),
    # Record index in which the hit was found
    # I assume nobody will process more than 2 147 483 647 pulses at a time
    ('record_i', np.int32),
]


def peak_dtype(n_channels, n_sum_wv_samples=200):
    """Data type for peaks - ranges across all channels in a detector
    Remember to set channel to -1 (todo: make enum)
    """
    return interval_dtype + [
        # Area per channel in ADC * samples
        ('area_per_channel', (np.int32, n_channels)),
        # Sum area in PE
        ('area', np.float32),
        # Number of hits from which this peak was constructed
        ('n_hits', np.int16),
        # Waveform data in PE/sample
        ('data', np.float32, n_sum_wv_samples),
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
            data = COMPRESSORS[compressor]['decompress'](f.read())
        # frombuffer is much faster, but results in some readonly fields
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
        d_comp = COMPRESSORS[compressor]['compress'](records)
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
    return filename + '.' + COMPRESSORS[compressor].get('extension',
                                                        compressor)
