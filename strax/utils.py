import numpy as np
import numba
import dill
from functools import wraps

__all__ = 'records_needed growing_result sort_by_time ' \
          'fully_contained_in'.split()

# Change numba's caching backend from pickle to dill
# I'm sure they don't mind...
# Otherwise we get strange errors while caching the @growing_result functions
numba.caching.pickle = dill


def records_needed(pulse_length, samples_per_record):
    """Return records needed to store pulse_length samples"""
    return 1 + (pulse_length - 1) // samples_per_record


def growing_result(dtype=np.int, chunk_size=10000):
    """Decorator factory for functions that fill numpy arrays
    Functions must obey following API:
     - accept _result_buffer keyword argument with default None;
       this will be the buffer array of specified dtype and length chunk_size
       (it's an optional argument so this decorator preserves signature)
     - 'yield N' from function will cause first elements to be saved
     - function is responsible for tracking offset, calling yield on time,
       and clearing the buffer afterwards.
     - optionally, accept result_dtype argument with default None;
       this allows function user to specify return dtype

    See test_utils.py for a simple example (I can't get it to run as a doctest
    unfortunately)
    """
    def _growing_result(f):
        @wraps(f)
        def wrapped_f(*args, **kwargs):

            if '_result_buffer' in kwargs:
                raise ValueError("_result_buffer argument is internal-only")

            _dtype = kwargs.get('result_dtype')
            if _dtype is None:
                _dtype = dtype
            else:
                del kwargs['result_dtype']   # Don't pass it on to function
            buffer = np.zeros(chunk_size, _dtype)

            # Keep a list of saved buffers to concatenate at the end
            saved_buffers = []
            for n_added in f(*args, _result_buffer=buffer, **kwargs):
                saved_buffers.append(buffer[:n_added].copy())

            # If nothing returned, return an empty array of the right dtype
            if not len(saved_buffers):
                return np.zeros(0, _dtype)
            return np.concatenate(saved_buffers)

        return wrapped_f

    return _growing_result


# (5-10x) faster than np.sort(order=...), as np.sort looks at all fields
# TODO: maybe this should be a factory?
@numba.jit(nopython=True, nogil=True, cache=True)
def sort_by_time(x):
    time = x['time'].copy()    # This increases speed even more
    sort_i = np.argsort(time)
    return x[sort_i]


@numba.jit(nopython=True, nogil=True, cache=True)
def fully_contained_in(things, containers):
    """Return array of len(things) with index of interval in containers
    for which things are fully contained in a container, or -1 if no such
    exists. We assume all intervals are sorted by time, and b_intervals
    nonoverlapping.
    """
    result = np.ones(len(things), dtype=np.int32) * -1
    a_starts = things['time']
    b_starts = containers['time']
    a_ends = a_starts + things['length'] * things['dt']
    b_ends = b_starts + containers['length'] * containers['dt']

    b_i = 0
    for a_i in range(len(things)):
        # Skip ahead one or more b's if we're beyond them
        while b_i < len(containers) and b_ends[b_i] < a_starts[a_i]:
            b_i += 1
        if b_i == len(containers):
            break

        if b_starts[b_i] <= a_starts[a_i] and a_ends[a_i] <= b_ends[b_i]:
            result[a_i] = b_i

    return result
