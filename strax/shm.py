"""Shared-memory numpy array passing using SharedArray

This code adds support for structured arrays, and a ProcessPoolExecutor
that auto-wraps the input/output arrays in temporary shared memory.
"""

import random
import pickle
import string
import numpy as np
import concurrent.futures

import strax
export, __all__ = strax.exporter()

try:
    import SharedArray
except ImportError as e:
    # Make strax useable on operating systems without SharedArray
    class SharedArrayMock:
        def __getattr__(self, name):
            def kaboom(*args, **kwargs):
                raise RuntimeError(
                    f"Attempt to use shared-memory message passing, "
                    f"but SharedArray did not import.")
            return kaboom
    SharedArray = SharedArrayMock()


def pack_dtype(dtype):
    """Pack dtype into a numpy array"""
    return np.frombuffer(pickle.dumps(strax.unpack_dtype(dtype)),
                         dtype=np.uint8)

def unpack_dtype(x):
    """Unpack dtype from a numpy array"""
    return np.dtype(pickle.loads(x))


@export
def shm_nuke():
    """Clear the entire shared memory"""
    for x in SharedArray.list():
        SharedArray.delete(x.name.decode())


@export
def shm_put(arr, temp=False, _key=None):
    """Put array into shared memory

    :param temp: If True, key will be deleted on its first retrieval by shm_pop
    (unless shm_pop is called with keep=True).
    :param _key: shared memory key to use. Do not set this yourself
    unless you know what you are doing.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Got {arr} ({type(arr)}) instead of a numpy array!")

    is_struct = arr.dtype.names is not None

    if _key is None:
        _key = 'shm://' + ':'.join([
            "sdata" if is_struct else "data",
            ''.join(random.choices(string.ascii_uppercase
                                   + string.digits,
                                   k=8)),
            "TEMP" if temp else ""])

    shared_arr = SharedArray.create(_key, arr.shape, dtype=arr.dtype)

    if is_struct:
        # Structured arrays can only be shared as raw void-dtype arrays
        shared_arr[:] = np.frombuffer(arr, dtype=np.void(arr.itemsize))

        # To recover the dtype later, we have to encode it as a separate array
        # and share that too
        shm_put(pack_dtype(arr.dtype),
                _key=_key.replace('sdata', 'dtype'),
                temp=temp)

    else:
        shared_arr[:] = arr[:]

    return _key

@export
def shm_pop(key, keep=None):
    """Return key from shm.

    :param keep: Whether to keep the shm or erase it.
    If not specified, will delete shm if key ends with '_TEMP'
    (e.g. it was created with temp=True to shm_put)
    """
    result = SharedArray.attach(key)

    arrtype, _, tmp = key[6:].split(':')

    is_temp = tmp == 'TEMP'
    if keep is None:
        keep = not is_temp

    if arrtype == "sdata":
        # This was a structured array. Recover the dtype:
        dtype = unpack_dtype(shm_pop(key.replace('sdata', 'dtype'),
                                     keep=keep))
        result = np.frombuffer(result, dtype=dtype)

    if not keep:
        shm_del(key)
    return result

@export
def shm_del(key):
    # Split of the 'shm://' prefix
    return SharedArray.delete(key[6:])


@export
def is_shmkey(key):
    return isinstance(key, str) and key.startswith('shm://')


def shm_wrap_f(f, *args, shm_out=True, **kwargs):
    # Get any shared memory args/kwargs
    for x in args:
        if isinstance(x, np.ndarray):
            raise ValueError(f"{f} got unshmd numpy arr in args!")
    for k, x in kwargs.items():
        if isinstance(x, np.ndarray):
            raise ValueError(f"{f} got unshmd numpy arr in kwargs {k}!")

    args = [shm_pop(x)
            if is_shmkey(x) else x
            for x in args]
    for k, v in kwargs.items():
        if is_shmkey(v):
            kwargs[k] = shm_pop(v)

    result = f(*args, **kwargs)

    if shm_out:
        if isinstance(result, np.ndarray):
            result = shm_put(result, temp=True)
        elif isinstance(result, dict):
            for k, v in result.items():
                if isinstance(v, np.ndarray):
                    result[k] = shm_put(v, temp=True)
    return result


@export
def unshm(x, keep=None):
    """Retrieve x from shared memory if it is a shared memory key
    If x is a dict, replace any shared memory keys in values by numpy arrays.
    """
    if is_shmkey(x):
        x = shm_pop(x, keep=keep)
    elif isinstance(x, dict):
        for k, v in x.items():
            if is_shmkey(v):
                x[k] = shm_pop(v, keep=keep)
    return x


@export
class SHMExecutor(concurrent.futures.ProcessPoolExecutor):
    """ProcessPoolExecutor that passes numpy arrays in and out of the
    job functions through shared memory, avoiding the pickling overhead
    in python multiprocessing.
    """

    def __init__(self, *args, **kwargs):
        shm_nuke()
        super().__init__(*args, **kwargs)

    def submit(self, f, *args, shm_input=True, shm_output=True, **kwargs):
        """Return future for f(*args, **kwargs) computation.

        :param shm_input: If True (default), transfer numpy array input
        through shared memory
        :param shm_output: If True (default), transfer output of f via shared
        memory if it is a numpy array (or dict containing possibly numpy
        arrays).
        NB: result will contain un-attached shared memory keys! Use
        strax.unshm to unpack the result.
        """
        if shm_input:
            # Copy numpy arguments to f to temporary shared memory
            args = [shm_put(x, temp=True)
                    if isinstance(x, np.ndarray) else x
                    for x in args]
            for k, v in kwargs.items():
                if isinstance(v, np.ndarray):
                    kwargs[k] = shm_put(v, temp=True)

        if shm_input or shm_output:
            return super().submit(shm_wrap_f, f, *args,
                                  shm_out=shm_output, **kwargs)
        else:
            return super().submit(f, *args, **kwargs)

##
# Patch base_future to unpack results from shm as soon as they are put
# This is dirty, but I don't know another safe way
# (e.g. adding callback is not OK, someone can get at the result
#  before the callback runs)
##

class Future(concurrent.futures._base.Future):

    def set_result(self, result):
        super().set_result(unshm(result))

concurrent.futures._base.Future = Future
