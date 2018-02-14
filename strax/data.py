"""Store pulses in memory-mapped numpy arrays with metadata
These slurp / dump very quickly  (I don't really understand why
they are so much faster than f.write(), np.save, etc)
"""
import bz2
import json
import mmap

import blosc
import zstd
import numpy as np
import numba

# Testing on 300 MB data, after baseline subtraction
# blosc 1000 MB/sec, 3x reduction
# zstd: 300 MB/sec, 5x reduction (harder to install?)
# bz2: 23 MB/sec, 6.4x reduction
# Other algorithms (zlib, snappy, lzma) are all worse than this
# (worse = takes longer AND reduces less)
COMPRESSORS = dict(bz2=bz2, blosc=blosc, zstd=zstd)


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


def save_metadata(filename, **metadata):
    with open(filename + '.json', mode='w') as f:
        f.write(json.dumps(dict(**metadata)))


def read_metadata(filename):
    with open(filename + '.json', mode='r') as f:
        metadata = json.loads(f.read())
    return metadata


def load_records(filename, with_meta=False):
    metadata = read_metadata(filename)
    # Read array from memory map file
    fp = np.memmap(filename, mode='r+', dtype=eval(metadata['dtype']))
    data = np.array(fp)
    del fp   # There is no close. Just being explicit.
    if with_meta:
        return data, metadata
    return data


def save_records(filename, records, **metadata):
    assert isinstance(records, np.ndarray), "Please pass a numpy array"
    save_metadata(filename,
                  dtype=records.dtype.descr.__repr__(),
                  **metadata)
    # Save array to memmap.
    fp = np.memmap(filename, mode='w+',
                   dtype=records.dtype, shape=records.shape)
    fp[:] = records[:]
    del fp   # There is no close. Just being explicit.
    return None


def save_records_compressed(filename, data, compressor='zstd', **metadata):
    save_metadata(filename,
                  compressor=compressor,
                  **metadata)

    # With shuffle=True, records w zeroed baselines take quite a lot of space
    # (compared to using the baseline instead of 0 as the filler value)
    # Perhaps related to twos-complement representation of signed integers?
    # shuffle=False + 0 for baseline gives the smallest file.
    shuffle_kwarg = dict(shuffle=False) if compressor == 'blosc' else dict()

    d_comp = COMPRESSORS[compressor].compress(data, **shuffle_kwarg)

    # Dump it to a memory-map file. Why is this so fast...
    with open(filename, "w+b") as f:
        mm = mmap.mmap(f.fileno(), len(d_comp))
        mm[:] = d_comp
        mm.close()


def load_records_compressed(filename):
    metadata = read_metadata(filename)
    compressor = COMPRESSORS[metadata['compressor']]
    raise NotImplementedError
