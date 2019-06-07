from base64 import b32encode
import collections
import contextlib
from functools import wraps
import json
import re
import sys
import traceback
import typing
from hashlib import sha1

import numpy as np
import numba
import dill


# Change numba's caching backend from pickle to dill
# I'm sure they don't mind...
# Otherwise we get strange errors while caching the @growing_result functions
numba.caching.pickle = dill


def exporter(export_self=False):
    """Export utility modified from https://stackoverflow.com/a/41895194
    Returns export decorator, __all__ list
    """
    all_ = []
    if export_self:
        all_.append('exporter')

    def decorator(obj):
        all_.append(obj.__name__)
        return obj

    return decorator, all_


export, __all__ = exporter(export_self=True)


@export
def inherit_docstring_from(cls):
    """Decorator for inheriting doc strings, stolen from
    https://groups.google.com/forum/#!msg/comp.lang.python/HkB1uhDcvdk/lWzWtPy09yYJ
    """
    def docstring_inheriting_decorator(fn):
        fn.__doc__ = getattr(cls, fn.__name__).__doc__
        return fn
    return docstring_inheriting_decorator


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
        def accumulate_numba_result(*args, **kwargs):

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
            if len(saved_buffers) == 1:
                return saved_buffers[0]
            else:
                return np.concatenate(saved_buffers)

        return accumulate_numba_result

    return _growing_result


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
def merged_dtype(dtypes):
    result = {}
    for x in dtypes:
        for unpacked_dtype in unpack_dtype(x):
            field_name = unpacked_dtype[0]
            if isinstance(field_name, tuple):
                field_name = field_name[1]
            if field_name in result:
                continue

            result[field_name] = unpacked_dtype

    return list(result.values())


@export
def merge_arrs(arrs):
    """Merge structured arrays of equal length.

    On field name collisions, data from later arrays is kept.
    """
    # Much faster than the similar function in numpy.lib.recfunctions

    n = len(arrs[0])
    if not all([len(x) == n for x in arrs]):
        raise ValueError("Arrays must all have the same length")

    result = np.zeros(n, dtype=merged_dtype([x.dtype for x in arrs]))
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


@export
@contextlib.contextmanager
def profile_threaded(filename):
    import yappi  # noqa   # yappi is not a dependency
    import gil_load  # noqa   # same
    yappi.set_clock_type("cpu")
    try:
        gil_load.init()
        gil_load.start(av_sample_interval=0.1,
                       output_interval=3,
                       output=sys.stdout)
        monitoring_gil = True
    except RuntimeError:
        monitoring_gil = False
        pass

    yappi.start()
    yield
    yappi.stop()

    if monitoring_gil:
        gil_load.stop()
        print("Gil was held %0.1f %% of the time" %
              (100 * gil_load.get()[0]))
    p = yappi.get_func_stats()
    p = yappi.convert2pstats(p)
    p.dump_stats(filename)
    yappi.clear_stats()


@export
def to_str_tuple(x) -> typing.Tuple[str]:
    if isinstance(x, str):
        return x,
    elif isinstance(x, list):
        return tuple(x)
    elif isinstance(x, tuple):
        return x
    raise TypeError(f"Expected string or tuple of strings, got {type(x)}")


@export
def hashablize(obj):
    """Convert a container hierarchy into one that can be hashed.
    See http://stackoverflow.com/questions/985294
    """
    try:
        hash(obj)
    except TypeError:
        if isinstance(obj, dict):
            return tuple((k, hashablize(v)) for (k, v) in sorted(obj.items()))
        elif isinstance(obj, np.ndarray):
            return tuple(obj.tolist())
        elif hasattr(obj, '__iter__'):
            return tuple(hashablize(o) for o in obj)
        else:
            raise TypeError("Can't hashablize object of type %r" % type(obj))
    else:
        return obj


@export
def deterministic_hash(thing, length=10):
    """Return a base32 lowercase string of length determined from hashing
    a container hierarchy
    """
    digest = sha1(json.dumps(hashablize(thing)).encode('ascii')).digest()
    return b32encode(digest)[:length].decode('ascii').lower()


@export
def formatted_exception():
    """Return human-readable multiline string with info
    about the exception that is currently being handled.

    If no exception, or StopIteration, is being handled,
    returns an empty string.

    For MailboxKilled exceptions, we return the original
    exception instead.
    """
    # Can't do this at the top level, utils is one of the
    # first files of the strax package
    import strax
    exc_info = sys.exc_info()
    if exc_info[0] == strax.MailboxKilled:
        # Get the original exception back out
        return '\n'.join(
            traceback.format_exception(*exc_info[1].args[0]))
    if exc_info[0] in [None, StopIteration]:
        # There was no relevant exception to record
        return ''
    return traceback.format_exc()


@export
def print_record(x, skip_array=True):
    """Print record(s) d in human-readable format
    :param skip_array: Omit printing array fields
    """
    if len(x.shape):
        for q in x:
            print_record(q)

    # Check what number of spaces required for nice alignment
    max_len = np.max([len(key) for key in x.dtype.names])
    for key in x.dtype.names:
        try:
            len(x[key])
        except TypeError:
            # Not an array field
            pass
        else:
            if skip_array:
                continue

        print(("{:<%d}: " % max_len).format(key), x[key])


@export
def count_tags(ds):
    """Return how often each tag occurs in the datasets DataFrame ds"""
    from collections import Counter
    from itertools import chain
    all_tags = chain(*[ts.split(',')
                       for ts in ds['tags'].values])
    return Counter(all_tags)


@export
def flatten_dict(d, separator=':', _parent_key=''):
    """Flatten nested dictionaries into a single dictionary,
    indicating levels by separator.
    Don't set _parent_key argument, this is used for recursive calls.
    Stolen from http://stackoverflow.com/questions/6027558
    """
    items = []
    for k, v in d.items():
        new_key = _parent_key + separator + k if _parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v,
                                      separator=separator,
                                      _parent_key=new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)