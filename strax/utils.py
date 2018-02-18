import numpy as np
import numba

__all__ = 'growing_result sort_by_time sort_by_channel_then_time'.split()


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
        def wrapped_f(*args, **kwargs):
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


# These sorting functions are considerably (5-10x) faster
# than np.sort with the order argument, as np.sort also looks at all fields
# in the array. Probably we should make some kind of factory for these instead
# of hardcoding channel and time.

# I chose to let these operate in-place as almost all strax functions
# are in-place. However, unless numba/llvm does some magic optimization,
# they do copy the data internally.

@numba.jit(nopython=True)
def sort_by_time(x):
    time = x['time'].copy()
    sort_i = np.argsort(time)
    x[:] = x[sort_i]


@numba.jit(nopython=True)
def sort_by_channel_then_time(x):
    result = np.zeros_like(x)

    # Determine indexes that would sort by channel
    channel = x['channel'].copy()
    channel_sort_order = np.argsort(channel)

    # Get channel and time, both sorted by channel
    channel = channel[channel_sort_order]
    times_sorted_by_channel = x['time'].copy()[channel_sort_order]

    start_i = 0
    n_in_ch = 1
    for i in range(1, len(channel)):
        if channel[i] != channel[i - 1]:
            # New range of channels
            end_i = start_i + n_in_ch

            result[start_i:end_i] = x[channel_sort_order[start_i:end_i][
                np.argsort(times_sorted_by_channel[start_i:end_i])]]

            start_i = i
            n_in_ch = 1
        else:
            n_in_ch += 1

    result[start_i:] = x[channel_sort_order[start_i:][
        np.argsort(times_sorted_by_channel[start_i:])]]

    x[:] = result
