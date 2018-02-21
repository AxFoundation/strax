import numpy as np
import numba

__all__ = 'growing_result sort_by_time fully_in_range'.split()


def growing_result(dtype=np.int, chunk_size=10000):
    """Decorator factory for functions that fill numpy arrays
    Functions must obey following API:
     - first argument is buffer array of specified dtype and length chunk_size
     - 'yield N' from function will cause first elements to be saved
     - function is responsible for tracking offset and calling yield on time!
    The buffer is not explicitly cleared.

    Silly example:
    >>> @growing_result(np.int, chunk_size=2)
    >>> def bla(buffer):
    >>>     offset = 0
    >>>     for i in range(5):
    >>>         buffer[offset] = i
    >>>         offset += 1
    >>>         if offset == len(buffer):
    >>>             yield offset
    >>>             offset = 0
    >>>     yield offset
    >>> bla()
    np.array([0, 1, 2, 3, 4])

    """
    def _growing_result(f):
        # Do not use functools.wraps, it messes with the signature
        # in ways numba does not appreciate
        # Note we allow user to override chunk_size and dtype after calling
        def wrapped_f(*args, chunk_size=chunk_size, dtype=dtype, **kwargs):
            buffer = np.zeros(chunk_size, dtype)

            # Keep a list of saved buffers to concatenate at the end
            saved_buffers = []
            for n_added in f(buffer, *args, **kwargs):
                saved_buffers.append(buffer[:n_added].copy())

            # If nothing returned, return an empty array of the right dtype
            if not len(saved_buffers):
                return np.zeros(0, dtype)
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
def fully_contained_in(a_intervals, b_intervals, dt=10):
    """Return array of len(a_intervals) with index of interval in b_intervals
    for which the interval a is fully contained, or -1 if no such exists.
    We assume all intervals are sorted by time, and b_intervals nonoverlapping.
    # TODO: OFF BY ONE TESTS!
    """
    result = np.zeros(len(a_intervals), dtype=np.bool_)
    a_starts = a_intervals['time']
    b_starts = b_intervals['time']
    a_ends = a_starts + a_intervals['length'] * a_intervals['dt']
    b_ends = b_starts+ b_intervals['length'] * b_intervals['dt']

    b_i = 0
    for a_i in range(len(a_intervals)):
        # Skip ahead one or more b's if we're beyond them
        while (b_i < len(b_intervals) and b_ends[b_i] < a_starts[a_i]):
            b_i += 1
        if b_i == len(b_intervals):
            break

        result[a_i] = (a_starts[a_i] >= b_starts[b_i]
                       and a_ends[a_i] <= b_ends[b_i])

    return result
