import numpy as np


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
            saved_buffers = []
            for n_added in f(buffer, *args, **kwargs):
                saved_buffers.append(buffer[:n_added].copy())
            return np.concatenate(saved_buffers)

        return wrapped_f

    return _growing_result
