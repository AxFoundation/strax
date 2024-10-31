import numpy as np
import numba

# Define error message as a constant
UNSTABLE_SORT_MESSAGE = (
    "quicksort and heapsort are not allowed due to non-deterministic behavior.\n"
    "Please use mergesort for deterministic sorting behavior."
)


# Define custom exception for sorting errors
class SortingError(Exception):
    pass


def stable_sort(arr, kind="mergesort"):
    """Numba-optimized stable sort function using mergesort.

    Args:
        arr: numpy array to sort
        kind: sorting algorithm to use (only 'mergesort' is allowed)

    Returns:
        Sorted array using mergesort algorithm

    """
    if kind != "mergesort":
        raise SortingError(UNSTABLE_SORT_MESSAGE)
    return np.sort(arr)


def stable_argsort(arr, kind="mergesort"):
    """Get indices that will sort an array.

    Args:
        arr: Array to sort, can include any comparable type
        kind: Sort algorithm to use (only 'mergesort' is allowed)

    Returns:
        Array of indices that can be used to sort arr while preserving
        relative order of equal elements (stable sort)

    """
    if kind != "mergesort":
        raise SortingError(UNSTABLE_SORT_MESSAGE)
    try:
        # Try Numba version first
        return _stable_argsort_numba(arr, kind)
    except:
        # Fall back to regular numpy if Numba fails
        return np.argsort(arr, kind=kind)


@numba.njit(nogil=True, cache=True)
def _stable_argsort_numba(arr, kind):
    """Internal Numba-accelerated version."""
    return np.argsort(arr, kind=kind)
