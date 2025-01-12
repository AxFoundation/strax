from base64 import b32encode
import collections
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import itertools
import contextlib
from functools import wraps
import json
import re
import sys
import traceback
import typing as ty
from hashlib import sha1
import strax
from strax import stable_argsort, stable_sort
import numexpr
import dill
import numba
import numpy as np
import pandas as pd
from collections.abc import Mapping
from warnings import warn
import os
import click
import deepdiff
import copy


# Change numba's caching backend from pickle to dill
# I'm sure they don't mind...
# Otherwise we get strange errors while caching the @growing_result functions
try:
    numba.core.caching.pickle = dill
except AttributeError:
    # Numba < 0.49
    numba.caching.pickle = dill

if any("jupyter" in arg for arg in sys.argv):
    # In some cases we are not using any notebooks,
    # Taken from 44952863 on stack overflow thanks!
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm  # type: ignore


def exporter(export_self=False):
    """Export utility modified from https://stackoverflow.com/a/41895194
    Returns export decorator, __all__ list
    """
    all_ = []
    if export_self:
        all_.append("exporter")

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
def growing_result(dtype=np.int64, chunk_size=10000):
    """Decorator factory for functions that fill numpy arrays.

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
            if "_result_buffer" in kwargs:
                raise ValueError("_result_buffer argument is internal-only")

            _dtype = kwargs.get("result_dtype")
            if _dtype is None:
                _dtype = dtype
            else:
                del kwargs["result_dtype"]  # Don't pass it on to function
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
def set_nan_defaults(buffer):
    """When constructing the dtype, take extra care to set values to np.Nan / -1 (for ints) as 0
    might have a meaning."""
    for field in buffer.dtype.names:
        if np.issubdtype(buffer.dtype[field], np.integer):
            buffer[field][:] = -1
        else:
            buffer[field][:] = np.nan


@export
def unpack_dtype(dtype):
    """Return list of tuples needed to construct the dtype.

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
def remove_titles_from_dtype(dtype):
    """Return np.dtype with titles removed from fields."""
    return np.dtype(
        [
            (fieldname[-1] if isinstance(fieldname, tuple) else fieldname, dt)
            for fieldname, dt in unpack_dtype(np.dtype(dtype))
        ]
    )


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
def merge_arrs(arrs, dtype=None, replacing=False):
    """Merge structured arrays of equal length. On field name collisions, data from later arrays is
    kept.

    replacing=True is usually used when you want to convert arrs into a new dtype

    If you pass one array, it is returned without copying unless replacing=True.

    Much faster than the similar function in numpy.lib.recfunctions.

    """
    if not len(arrs):
        raise RuntimeError("Cannot merge 0 arrays")
    if len(arrs) == 1 and not replacing:
        return arrs[0]

    n = len(arrs[0])
    if not all([len(x) == n for x in arrs]):
        raise ValueError(
            "Arrays to merge must have the same length, got lengths "
            + ", ".join([str(len(x)) for x in arrs])
        )

    if dtype is None:
        dtype = merged_dtype([x.dtype for x in arrs])

    result = np.zeros(n, dtype=dtype)
    for arr in arrs:
        for fn in arr.dtype.names:
            if fn in result.dtype.names:
                result[fn] = arr[fn]
    return result


@export
def camel_to_snake(x):
    """Convert x from CamelCase to snake_case."""
    # From https://stackoverflow.com/questions/1175208
    x = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", x)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", x).lower()


@export
@contextlib.contextmanager
def profile_threaded(filename):
    import yappi  # noqa   # yappi is not a dependency

    yappi.set_clock_type("cpu")
    try:
        import gil_load  # noqa   # same

        gil_load.init()
        gil_load.start(av_sample_interval=0.1, output_interval=10, output=sys.stdout)
        monitoring_gil = True
    except (RuntimeError, ImportError):
        monitoring_gil = False

    yappi.start()
    yield
    yappi.stop()

    if monitoring_gil:
        gil_load.stop()
        stats = gil_load.get()
        print("GIL load information: ", gil_load.format(stats))
    p = yappi.get_func_stats()
    p = yappi.convert2pstats(p)
    p.dump_stats(filename)
    yappi.clear_stats()


@export
def to_str_tuple(x) -> ty.Tuple[ty.Any, ...]:
    if isinstance(x, (str, bytes)):
        return (x,)
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
    if isinstance(obj, Mapping):
        # Convert immutabledict etc for json decoding
        obj = dict(obj)
    try:
        hash(obj)
    except TypeError:
        if isinstance(obj, dict):
            return tuple((k, hashablize(v)) for (k, v) in sorted(obj.items()))
        elif isinstance(obj, np.ndarray):
            return tuple(obj.tolist())
        elif hasattr(obj, "__iter__"):
            return tuple(hashablize(o) for o in obj)
        else:
            raise TypeError("Can't hashablize object of type %r" % type(obj))
    else:
        return obj


@export
class NumpyJSONEncoder(json.JSONEncoder):
    """Special json encoder for numpy types
    Edited from mpl3d: mpld3/_display.py
    """

    def default(self, obj):
        try:
            iterable = iter(obj)
        except TypeError:
            pass
        else:
            return [self.default(item) for item in iterable]
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@export
def deterministic_hash(thing, length=10):
    """Return a base32 lowercase string of length determined from hashing a container hierarchy."""
    hashable = hashablize(thing)
    jsonned = json.dumps(hashable, cls=NumpyJSONEncoder)
    # disable bandit
    digest = sha1(jsonned.encode("ascii")).digest()
    return b32encode(digest)[:length].decode("ascii").lower()


@export
def compare_dict(old: dict, new: dict):
    """Compare two dictionaries and print the differences."""
    differences = deepdiff.DeepDiff(old, new)
    color_values = lambda oldval, newval: (
        click.style(oldval, fg="red", bold=True),
        click.style(newval, fg="green", bold=True),
    )
    underline = lambda text, bold=True: click.style(text, bold=bold, underline=True)
    for key, value in differences.items():
        if key in ["values_changed", "iterable_item_added", "iterable_item_removed"]:
            print(underline(f"\n> {key}"))
            for kk, vv in value.items():
                if key == "values_changed":
                    old_values = vv["old_value"]
                    new_values = vv["new_value"]
                elif key == "iterable_item_added":
                    old_values = "-"
                    new_values = vv
                else:  # if key == "iterable_item_removed":
                    old_values = vv
                    new_values = "-"
                old, new = color_values(old_values, new_values)
                click.secho(f"\t in {kk[4:]}", bold=False)
                print(f"\t\t{old} -> {new}")
        elif key in ["dictionary_item_added", "dictionary_item_removed"]:
            color = "red" if "removed" in key else "green"
            print(underline(f"\n> {key:25s}"), end="->")
            click.secho(f"\t{', '.join(value)}", fg=color)
        elif key in ["type_changes"]:
            print(underline(f"\n> {key}"))
            for kk, vv in value.items():
                click.secho(f"\t{kk}")
                oldtype = vv["old_type"]
                newtype = vv["new_type"]
                keyold, keynew = color_values("old_type", "new_type")
                valueold, valuenew = color_values(vv["old_value"], vv["new_value"])
                print(f"\t\t{keyold:10s} : {oldtype} ({valueold})")
                print(f"\t\t{keynew:10s} : {newtype} ({valuenew})")
        else:
            raise KeyError(f"Unkown key in comparison {key}")


@export
def formatted_exception():
    """Return human-readable multiline string with info about the exception that is currently being
    handled.

    If no exception, or StopIteration, is being handled, returns an empty string.

    For MailboxKilled exceptions, we return the original exception instead.

    """
    exc_info = sys.exc_info()
    if exc_info[0] == strax.MailboxKilled:
        # Get the original exception back out
        return "\n".join(traceback.format_exception(*exc_info[1].args[0]))
    if exc_info[0] in [None, StopIteration]:
        # There was no relevant exception to record
        return ""
    return traceback.format_exc()


@export
def print_record(x, skip_array=True):
    """Print record(s) d in human-readable format.

    :param skip_array: Omit printing array fields.

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
    """Return how often each tag occurs in the datasets DataFrame ds."""
    from collections import Counter
    from itertools import chain

    all_tags = chain(*[ts.split(",") for ts in ds["tags"].values])
    return Counter(all_tags)


@export
def flatten_dict(d, separator=":", _parent_key="", keep=tuple()):
    """Flatten nested dictionaries into a single dictionary, indicating levels by separator.

    Don't set _parent_key argument, this is used for recursive calls. Stolen from
    http://stackoverflow.com/questions/6027558
    :param keep: key or list of keys whose values should not be flattened.

    """
    keep = to_str_tuple(keep)
    items = []
    for k, v in d.items():
        new_key = _parent_key + separator + k if _parent_key else k
        if isinstance(v, collections.abc.MutableMapping) and k not in keep:
            items.extend(flatten_dict(v, separator=separator, _parent_key=new_key).items())
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
            dtype.append((x, float))
        else:
            raise ValueError(f"Invalid field specification {x}")
    return np.dtype(dtype)


@export
def dict_to_rec(x, dtype=None):
    """Convert dictionary {field_name: array} to record array Optionally, provide dtype."""
    if isinstance(x, np.ndarray):
        return x

    if dtype is None:
        if not len(x):
            raise ValueError("Cannot infer dtype from empty dict")
        dtype = to_numpy_dtype([(k, np.asarray(v).dtype) for k, v in x.items()])

    if not len(x):
        return np.empty(0, dtype=dtype)

    some_key = list(x.keys())[0]
    n = len(x[some_key])
    r = np.zeros(n, dtype=dtype)
    for k, v in x.items():
        r[k] = v
    return r


@export
def multi_run(
    exec_function,
    run_ids,
    *args,
    max_workers=None,
    throw_away_result=False,
    multi_run_progress_bar=True,
    ignore_errors=False,
    log=None,
    **kwargs,
):
    """Execute exec_function(run_id, *args, **kwargs) over multiple runs, then return list of result
    arrays, each with a run_id column added.

    :param exec_function: Function to run
    :param run_ids: list/tuple of run_ids
    :param max_workers: number of worker threads/processes to spawn. If set to None, defaults to 1.
    :param throw_away_result: instead of collecting result, return None.
    :param multi_run_progress_bar: show a tqdm progressbar for multiple runs.
    :param ignore_errors: Return the data for the runs that successfully loaded, even if some runs
        failed executing.
    :param log: logger to be used.   Other (kw)args will be passed to the exec_function.

    """
    if max_workers is None:
        max_workers = 1

    if log is None:
        import logging

        log = logging.getLogger("strax_multi_run")
    # Only schedule twice as many tasks as there are workers. In this
    # way we avoid an overload of memory due to too many runs
    # (scales with number of runs)
    how_many_tasks_at_once = max_workers * 2
    task_index = 0

    # This will autocast all run ids to Unicode fixed-width
    run_id_numpy = np.array(run_ids)
    run_id_numpy = stable_sort(run_id_numpy)
    _is_superrun = np.any([r.startswith("_") for r in run_id_numpy])

    # Get from kwargs whether output should contain a run_id field.
    # In case we have a multi-runs with superruns we should skip adding
    # run_ids and sorting according run_id does not make sense.
    # (Have to delete it from kwargs to make not a new context later on)
    add_run_id_field = kwargs.setdefault("add_run_id_field", not _is_superrun)
    del kwargs["add_run_id_field"]
    run_id_as_bytes = kwargs.setdefault("run_id_as_bytes", False)
    del kwargs["run_id_as_bytes"]

    _add_run_id_as_byte = add_run_id_field and run_id_as_bytes
    if not _add_run_id_as_byte and len(run_id_numpy) > 70:
        warn(
            "You are asking for more than 70 runs at a time with add_run_id_field=True. "
            "Changing run_id data_type from string to bytes would reduce memory consumption. "
            'Do so with passing "run_id_as_bytes=True" . When you do, '
            'please note that "run_id" != b"run_id"! You can convert a byte string back to '
            'a normal string via b"byte_string".decode("utf-8"). '
        )
    elif _add_run_id_as_byte:
        run_id_numpy = run_id_numpy.astype("S")  # Use byte string to reduce memory usage.

    # List to sort data in the end according to output
    # (order may change due to threads)
    run_id_output = []

    # Generally we don't want a per run pbar because of multi_run_progress_bar
    kwargs.setdefault("progress_bar", False)

    # Probably we'll want to use dask for this in the future,
    # to enable cut history tracking and multiprocessing.
    # For some reason the ProcessPoolExecutor doesn't work??
    pbar = tqdm(
        total=len(run_id_numpy),
        desc="Loading %d runs" % len(run_ids),
        disable=not multi_run_progress_bar,
    )
    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as exc:
        log.debug("Starting ThreadPoolExecutor for multi-run.")
        # Submit first bunch of futures, add additional futures later
        futures = {
            exc.submit(exec_function, r, *args, **kwargs): r
            for r in itertools.islice(run_id_numpy, task_index, how_many_tasks_at_once)
        }

        task_index = how_many_tasks_at_once
        tasks_done = 0
        log.debug(f"Submitting first futures: {futures.values()}")
        final_result = []
        while futures:
            futures_done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for f in futures_done:
                tasks_done += 1
                _run_id = futures.pop(f)
                log.debug(
                    f"Done with run_id: {_run_id} and {len(run_id_numpy) - tasks_done} are left."
                )
                pbar.update(1)
                if f.exception() is not None:
                    if ignore_errors:
                        log.warning(f"Ran into {f.exception()}, ignoring that for now!")
                        failures.append(_run_id)
                        continue
                    raise f.exception()

                if throw_away_result:
                    continue
                result = f.result()

                # Append the run id column
                if add_run_id_field:
                    ids = np.array([_run_id] * len(result), dtype=[("run_id", run_id_numpy.dtype)])
                    result = merge_arrs([ids, result])
                final_result.append(result)
                run_id_output.append(_run_id)

            for r in itertools.islice(run_id_numpy, task_index, task_index + len(futures_done)):
                task_index += 1
                fut = exc.submit(exec_function, r, *args, **kwargs)
                futures[fut] = r
                log.debug(f"Submitting additional futures, new futures are: {futures.values()}")

        if throw_away_result:
            pbar.close()
            return None

        final_result = [final_result[ind] for ind in stable_argsort(run_id_output)]
        pbar.close()
        if ignore_errors and len(failures):
            log.warning(
                f"Failures for {len(failures) / len(run_ids):.0%} of runs. Failed for: {failures}"
            )
        return final_result


@export
def group_by_kind(dtypes, plugins=None, context=None) -> ty.Dict[str, ty.List]:
    """Return dtypes grouped by data kind i.e. {kind1: [d, d, ...], kind2: [d, d, ...], ...}

    :param plugins: plugins providing the dtypes.
    :param context: context to get plugins from if not given.

    """
    if plugins is None:
        if context is None:
            raise RuntimeError("group_by_kind requires plugins or context")
        plugins = context._get_plugins(targets=dtypes, run_id="0")

    deps_by_kind: ty.Dict = dict()
    for d in dtypes:
        p = plugins[d]
        k = p.data_kind_for(d)
        deps_by_kind.setdefault(k, [])
        deps_by_kind[k].append(d)

    return deps_by_kind


@export
def iter_chunk_meta(md):
    """Iterate over chunk info from metadata md adding n_from and n_to fields."""
    _n_to = _n_from = 0
    for c in md["chunks"]:
        _n_from = _n_to
        _n_to = _n_from + c["n"]
        c["n_from"] = _n_from
        c["n_to"] = _n_to
        yield c


@export
def parse_selection(x, selection):
    """Parse a selection string into a mask that can be used to filter data.

    :param selection: Query string, sequence of strings, or simple function to apply.
    :return: Boolean indicating the selected items.

    """
    if hasattr(selection, "__call__"):
        mask = selection(x)
    else:
        if isinstance(selection, (list, tuple)):
            selection = " & ".join(f"({x})" for x in selection)

        mask = numexpr.evaluate(selection, local_dict={fn: x[fn] for fn in x.dtype.names})
    return mask


@export
def apply_keep_columns(x, keep_columns):
    # Construct the new dtype
    new_dtype = []
    fields_to_copy = []
    for unpacked_dtype in strax.unpack_dtype(x.dtype):
        field_name = unpacked_dtype[0]
        if isinstance(field_name, tuple):
            field_name = field_name[1]

        if field_name in keep_columns:
            new_dtype.append(unpacked_dtype)
            fields_to_copy.append(field_name)

    # Copy over the data
    x2 = np.zeros(len(x), dtype=new_dtype)
    for field_name in keep_columns:
        x2[field_name] = x[field_name]
    return x2


@export
def apply_selection(
    x,
    selection=None,
    keep_columns=None,
    drop_columns=None,
    time_range=None,
    time_selection="fully_contained",
):
    """Return x after applying selections.

    :param x: Numpy structured array
    :param selection: Query string, sequence of strings, or simple function to apply.
    :param time_range: (start, stop) range to load, in ns since the epoch
    :param keep_columns: Field names of the columns to keep.
    :param drop_columns: Field names of the columns to drop.
    :param time_selection: Kind of time selectoin to apply:
        - skip: Do not select a time range, even if other arguments say so
        - touching: select things that (partially) overlap with the range
        - fully_contained: (default) select things fully contained in the range

    The right bound is, as always in strax, considered exclusive.
    Thus, data that ends (exclusively) exactly at the right edge of a
    fully_contained selection is returned.

    """
    if drop_columns and keep_columns:
        raise ValueError(
            "You cannot specify both keep_columns and drop_columns "
            "as their logic is contradictory please specify just one."
        )

    # Apply the time selections
    if time_range is None or time_selection == "skip":
        pass
    elif time_selection == "fully_contained":
        x = x[(time_range[0] <= x["time"]) & (strax.endtime(x) <= time_range[1])]
    elif time_selection == "touching":
        x = x[(strax.endtime(x) > time_range[0]) & (x["time"] < time_range[1])]
    else:
        raise ValueError(f"Unknown time_selection {time_selection}")

    if selection:
        x = x[parse_selection(x, selection)]

    if keep_columns:
        keep_columns = strax.to_str_tuple(keep_columns)

    if drop_columns:
        drop_columns = strax.to_str_tuple(drop_columns)
        keep_columns = []
        for unpacked_dtype in strax.unpack_dtype(x.dtype):
            field_name = unpacked_dtype[0]
            if isinstance(field_name, tuple):
                field_name = field_name[1]

            if field_name in drop_columns:
                continue
            keep_columns.append(field_name)

    if keep_columns:
        # We check before if keep and drop are specified both,
        # if so we raise an error.
        x2 = apply_keep_columns(x, keep_columns)
        x = x2
        del x2

    return x


def dir_size_mb(start_path="."):
    """
    Calculate the total size of all files in start_path
    Thanks https://stackoverflow.com/a/1392549/18280620
    """
    if not os.path.exists(start_path):
        return 0

    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size / 1e6


@export
def convert_tuple_to_list(init_func_input):
    """Convert the tuples into list in an arbitrarily nested dictionary."""
    func_input = copy.deepcopy(init_func_input)
    # if it is a tuple convert it and reiterate
    if isinstance(func_input, tuple):
        _func_input = list(func_input)
        return convert_tuple_to_list(_func_input)

    # if it is a list, go over all the elements until all tuples are lists
    elif isinstance(func_input, list):
        new_func_inp = []
        # check each element
        for i in func_input:
            new_func_inp.append(convert_tuple_to_list(i))
            # iterates until everything is all depths are exhausted
        return new_func_inp

    # if it is a dict iterate
    elif isinstance(func_input, dict):
        for k, v in func_input.items():
            func_input[k] = convert_tuple_to_list(v)
        return func_input
    else:
        # if not a container, return. i.e. int, float, bytes, str etc.
        return func_input


@export
def convert_structured_array_to_df(structured_array, log=None):
    """Convert a structured numpy array to a pandas DataFrame.

    Parameters:
    structured_array (numpy.ndarray): The structured array to be converted.
    Returns:
    pandas.DataFrame: The converted DataFrame.

    """

    if log is None:
        import logging

        log = logging.getLogger("strax_array_to_df")

    data_dict = {}
    converted_cols = []
    for name in structured_array.dtype.names:
        col = structured_array[name]
        if col.ndim > 1:
            # Convert n-dimensional columns to lists of ndarrays
            data_dict[name] = [np.array(row) for row in col]
            converted_cols.append(name)
        else:
            data_dict[name] = col

    if converted_cols:
        log.warning(
            f"Columns {converted_cols} contain non-scalar entries. "
            "Some pandas functions (e.g., groupby, apply) might "
            "not perform as expected on these columns."
        )

    return pd.DataFrame(data_dict)
