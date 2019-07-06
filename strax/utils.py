from base64 import b32encode
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
from functools import wraps
import json
import re
import sys
import traceback
import typing as ty
from hashlib import sha1

import dill
import numba
import numpy as np
from tqdm import tqdm
import pandas as pd

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

    If you pass one array, it is returned without copying.
    TODO: hmm... inconsistent

    Much faster than the similar function in numpy.lib.recfunctions.
    """
    if not len(arrs):
        raise RuntimeError("Cannot merge 0 arrays")
    if len(arrs) == 1:
        return arrs[0]

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
def to_str_tuple(x) -> ty.Tuple[str]:
    if isinstance(x, str):
        return x,
    elif isinstance(x, list):
        return tuple(x)
    elif isinstance(x, tuple):
        return x
    elif isinstance(x, pd.Series):
        return tuple(x.values.tolist())
    elif isinstance(x, np.ndarray):
        return tuple(x.tolist())
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


@export
def to_numpy_dtype(field_spec):
    if isinstance(field_spec, np.dtype):
        return field_spec

    dtype = []
    for x in field_spec:
        if len(x) == 3:
            if isinstance(x[0], tuple):
                # Numpy syntax for array field
                dtype.append(x)
            else:
                # Lazy syntax for normal field
                field_name, field_type, comment = x
                dtype.append(((comment, field_name), field_type))
        elif len(x) == 2:
            # (field_name, type)
            dtype.append(x)
        elif len(x) == 1:
            # Omitted type: assume float
            dtype.append((x, np.float))
        else:
            raise ValueError(f"Invalid field specification {x}")
    return np.dtype(dtype)


@export
def dict_to_rec(x, dtype=None):
    """Convert dictionary {field_name: array} to record array
    Optionally, provide dtype
    """
    if isinstance(x, np.ndarray):
        return x

    if dtype is None:
        if not len(x):
            raise ValueError("Cannot infer dtype from empty dict")
        dtype = to_numpy_dtype([(k, np.asarray(v).dtype)
                                for k, v in x.items()])

    if not len(x):
        return np.empty(0, dtype=dtype)

    some_key = list(x.keys())[0]
    n = len(x[some_key])
    r = np.zeros(n, dtype=dtype)
    for k, v in x.items():
        r[k] = v
    return r


@export
def multi_run(f, run_ids, *args, max_workers=None, **kwargs):
    """Execute f(run_id, **kwargs) over multiple runs,
    then return list of results.

    :param run_ids: list/tuple of runids
    :param max_workers: number of worker threads/processes to spawn

    Other (kw)args will be passed to f
    """
    # Try to int all run_ids

    # Get a numpy array of run ids.
    try:
        run_id_numpy = np.array([int(x) for x in run_ids],
                                dtype=np.int32)
    except ValueError:
        # If there are string id's among them,
        # numpy will autocast all the run ids to Unicode fixed-width
        run_id_numpy = np.array(run_ids)

    # Probably we'll want to use dask for this in the future,
    # to enable cut history tracking and multiprocessing.
    # For some reason the ProcessPoolExecutor doesn't work??
    with ThreadPoolExecutor(max_workers=max_workers) as exc:
        futures = [exc.submit(f, r, *args, **kwargs)
                   for r in run_ids]
        for _ in tqdm(as_completed(futures),
                      desc="Loading %d runs" % len(run_ids)):
            pass

        result = []
        for i, f in enumerate(futures):
            r = f.result()
            ids = np.array([run_id_numpy[i]] * len(r),
                           dtype=[('run_id', run_id_numpy.dtype)])
            r = merge_arrs([ids, r])
            result.append(r)
        return result


@export
def group_by_kind(dtypes, plugins=None, context=None,
                  require_time=None) -> ty.Dict[str, ty.List]:
    """Return dtypes grouped by data kind
    i.e. {kind1: [d, d, ...], kind2: [d, d, ...], ...}
    :param plugins: plugins providing the dtypes.
    :param context: context to get plugins from if not given.
    :param require_time: If True, one data type of each kind
    must provide time information. It will be put first in the list.

    If require_time is None (default), we will require time only if there
    is more than one data kind in dtypes.
    """
    if plugins is None:
        if context is None:
            raise RuntimeError("group_by_kind requires plugins or context")
        plugins = context._get_plugins(targets=dtypes, run_id='0')

    if require_time is None:
        require_time = len(group_by_kind(
            dtypes, plugins=plugins, context=context, require_time=False)) > 1

    deps_by_kind = dict()
    key_deps = []
    for d in dtypes:
        p = plugins[d]
        k = p.data_kind_for(d)
        deps_by_kind.setdefault(k, [])

        # If this has time information, put it first in the list
        if (require_time
                and 'time' in p.dtype_for(d).names):
            key_deps.append(d)
            deps_by_kind[k].insert(0, d)
        else:
            deps_by_kind[k].append(d)

    if require_time:
        for k, d in deps_by_kind.items():
            if not d[0] in key_deps:
                raise ValueError(f"No dependency of data kind {k} "
                                 "has time information!")

    return deps_by_kind
