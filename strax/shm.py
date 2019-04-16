"""Shared-memory numpy array passing
using SharedArray
"""

import random
import zstd
import string
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import strax
export, __all__ = strax.exporter()

try:
    import SharedArray
except ImportError as e:
    class SharedArray(object):
        def __getattr__(self, name):
            def kaboom(*args, **kwargs):
                raise RuntimeError(
                    f"Attempt to use shared-memory message passing, "
                    f"but SharedArray did not import: {e}")
            return kaboom


magic_offset = 192

def dtype_to_fnkey(dt):

    try:
        # Remove titles field
        compact_dtype = [(x[0][1], x[1])
                         for x in strax.unpack_dtype(dt)]
        compact_dtype = np.dtype(compact_dtype)
    except:
        compact_dtype = dt
    # Get compact string
    q = str(compact_dtype)
    for x in "'()'[]< ":
        q = q.replace(x, '')
    q = zstd.compress(q.encode('ascii'))
    # Encode as legal filename
    # After 192 we get reasonable characters
    return ''.join([chr(magic_offset + x)
                    for x in np.frombuffer(q, dtype=np.uint8)])

def fnkey_to_dtype(q):
    q = np.array([ord(y) - magic_offset for y in q], dtype=np.uint8)
    q = zstd.decompress(bytes(memoryview(q)))
    fields = q.decode('ascii').split(',')
    dtype = []
    i = 0
    while i < len(fields) - 1:
        try:
            int(fields[i + 2])
            dtype.append((fields[i], (fields[i+1], int(fields[i+2]))))
            i += 4

        except:
            dtype.append((fields[i], fields[i+1]))
            i += 2
    print(dtype)
    return np.dtype(dtype)


@export
def shm_nuke():
    for x in SharedArray.list():
        SharedArray.delete(x.name.decode())


@export
def shm_put(arr, temp=False):
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Got {arr} ({type(arr)}) instead of a numpy array!")

    key = 'shm://'

    is_struct = False
    if arr.dtype.names is not None:
        is_struct = True

        # Structured arrays are special: sharedarray will store them as
        # raw void-dtype arrays
        # To recover the type later, we have to encode the dtype in they key...
        key += f'struct_[{dtype_to_fnkey(arr.dtype)[:120]}]_'

    key += ''.join(random.choices(string.ascii_lowercase + string.digits,
                                  k=8))
    if temp:
        key += '_TEMP'
    shared_arr = SharedArray.create(key, arr.shape, dtype=arr.dtype)
    if is_struct:
        shared_arr[:] = np.frombuffer(arr, dtype=np.void(arr.itemsize))
    else:
        shared_arr[:] = arr[:]

    return key

@export
def shm_pop(key, keep=None):
    """Return key from shm.
    :param keep: Whether to keep the shm or erase it.
    If not specified, will delete shm if key ends with '_TEMP'
    (e.g. it was created with temp=True to shm_put)
    """
    result = SharedArray.attach(key)
    print(key)
    if 'struct_' in key:
        # Recover array structure from key...
        result = np.frombuffer(
            result,
            dtype=fnkey_to_dtype(key.split('[')[1].split(']')[0]))

    if keep is None:
        keep = not key.endswith('_TEMP')
    if not keep:
        shm_del(key)
    return result

@export
def shm_del(key):
    return SharedArray.delete(key[6:])


@export
def is_shmkey(key):
    return isinstance(key, str) and key.startswith('shm://')


def shm_wrap_f(f, *args, **kwargs):
    # Get any shared memory args/kwargs
    args = [shm_pop(x)
            if is_shmkey(x) else x
            for x in args]
    for k, v in kwargs.items():
        if is_shmkey(k):
            kwargs[k] = shm_pop(v)

    result = f(*args, **kwargs)

    if isinstance(result, np.ndarray):
        return shm_put(result, temp=True)
    elif isinstance(result, dict):
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                result[k] = shm_put(v, temp=True)
        return result
    return result


@export
def unshm(x, keep=None):
    if is_shmkey(x):
        x = shm_pop(x, keep=keep)
    elif isinstance(x, dict):
        for k, v in x.items():
            if is_shmkey(v):
                x[k] = shm_pop(v, keep=keep)
    return x


@export
class SHMExecutor(ProcessPoolExecutor):
    """ProcessPoolExecutor that passes numpy arrays through shared memory
    """

    # TODO: Results cannot be auto-unshmed yet, I think that would involve
    # a deeper hack ofProcessPoolExecutor
    # NB: adding a callback for this does NOT work,
    # someone else might get to the future before the callback runs.

    def __init__(self, *args, **kwargs):
        shm_nuke()
        super().__init__(*args, **kwargs)

    def submit(self, f, *args, shm_input=True, auto_unshm=True, **kwargs):
        if shm_input:
            # Copy numpy arguments to f to temporary shared memory
            args = [shm_put(x, temp=True)
                    if isinstance(x, np.ndarray) else x
                    for x in args]
            for k, v in kwargs.items():
                if isinstance(k, np.ndarray):
                    kwargs[k] = shm_put(v, temp=True)

        if auto_unshm:
            return super().submit(shm_wrap_f, f, *args, **kwargs)
        else:
            return super().submit(f, *args, **kwargs)
