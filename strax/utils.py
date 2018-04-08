from functools import wraps
import re

import numpy as np
import numba
import dill

# Change numba's caching backend from pickle to dill
# I'm sure they don't mind...
# Otherwise we get strange errors while caching the @growing_result functions
numba.caching.pickle = dill


def exporter(export_self=False):
    """Export utility modified from https://stackoverflow.com/a/41895194"""
    all_ = []
    if export_self:
        all_.append('exporter')

    def decorator(obj):
        all_.append(obj.__name__)
        return obj

    return decorator, all_


export, __all__ = exporter(export_self=True)


@export
def records_needed(pulse_length, samples_per_record):
    """Return records needed to store pulse_length samples"""
    return 1 + (pulse_length - 1) // samples_per_record


@export
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
@export
@numba.jit(nopython=True, nogil=True, cache=True)
def sort_by_time(x):
    """Sort pulses by time, then channel.

    Assumes you have no more than 10k channels, and records don't span
    more than 100 days. TODO: FIX this
    """
    # I couldn't get fast argsort on multiple keys to work in numba
    # So, let's make a single key...
    sort_key = (x['time'] - x['time'].min()) * 10000 + x['channel']
    sort_i = np.argsort(sort_key)
    return x[sort_i]


@export
@numba.jit(nopython=True, nogil=True, cache=True)
def first_index_not_below(arr, t):
    """Return first index of array >= t, or len(arr) if no such found"""
    for i, x in enumerate(arr):
        if x >= t:
            return i
    return len(arr)


@export
def endtime(x):
    """Return endtime of intervals x"""
    if 'endtime' in x.dtype.names:
        return x['endtime']
    return x['time'] + x['length'] * x['dt']


@export
def fully_contained_in(things, containers):
    """Return array of len(things) with index of interval in containers
    for which things are fully contained in a container, or -1 if no such
    exists. We assume all intervals are sorted by time, and b_intervals
    nonoverlapping.
    """
    result = np.ones(len(things), dtype=np.int32) * -1
    a_starts = things['time']
    b_starts = containers['time']
    a_ends = endtime(things)
    b_ends = endtime(containers)
    _fc_in(a_starts, b_starts, a_ends, b_ends, result)
    return result


@numba.jit(nopython=True, nogil=True, cache=True)
def _fc_in(a_starts, b_starts, a_ends, b_ends, result):
    b_i = 0
    for a_i in range(len(a_starts)):
        # Skip ahead one or more b's if we're beyond them
        # Note <= in second condition: end is an exclusive bound
        while b_i < len(b_starts) and b_ends[b_i] <= a_starts[a_i]:
            b_i += 1
        if b_i == len(b_starts):
            break

        # Check for containment. We only need to check one b, since bs
        # are nonoverlapping
        if b_starts[b_i] <= a_starts[a_i] and a_ends[a_i] <= b_ends[b_i]:
            result[a_i] = b_i


@export
def split_by_containment(things, containers):
    """Return list of thing-arrays contained in each container

    Assumes everything is sorted, and containers are nonoverlapping
    TODO: needs tests!
    """
    if not len(containers):
        return []

    which_container = fully_contained_in(things, containers)

    mask = which_container != -1
    things = things[mask]
    which_container = which_container[mask]
    things_split = np.split(
        things,
        np.where(np.diff(which_container))[0] + 1)

    # Insert empties for containers with nothing
    for c in np.setdiff1d(np.arange(len(containers)),
                          np.unique(which_container)):
        if c == 0:
            continue   # np.split already produces an empty in this case?
        things_split.insert(c, things[:0])

    return things_split


@export
def unpack_dtype(dtype):
    """Return list of tuples needed to construct the dtype

    dtype == np.dtype(unpack_dtype(dtype))
    """
    result = []
    fields = dtype.fields
    for field_name in dtype.names:
        fieldinfo = fields[field_name]
        if len(fieldinfo) == 3:
            # The field has a "titles" attribute.
            # In this case, the tuple returned by .fields is inconsistent
            # with the tuple expected by np.dtype constructor :-(
            field_dtype, some_number, field_title = fieldinfo
            result.append(((field_title, field_name), field_dtype))
        else:
            field_dtype, some_number = fieldinfo
            result.append((field_name, field_dtype))
    return result


@export
def merge_arrs(arrs):
    """Merge structured arrays of equal length. Assumes no field collisions.
    """
    n = len(arrs[0])
    if not all(np.array([len(x) for x in arrs]) == n):
        raise ValueError("Arrays must all have the same length")
    result_dtype = sum([unpack_dtype(x.dtype) for x in arrs], [])
    result = np.zeros(n, dtype=result_dtype)
    for arr in arrs:
        for fn in arr.dtype.names:
            result[fn] = arr[fn]
    return result


@export
def camel_to_snake(x):
    """Convert x from CamelCase to snake_case"""
    # From https://stackoverflow.com/questions/1175208
    x = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', x)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', x).lower()
