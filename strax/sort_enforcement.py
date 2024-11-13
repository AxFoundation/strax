import numpy as np
from numba.extending import register_jitable

# Define error message as a constant
UNSTABLE_SORT_MESSAGE = (
    "quicksort and heapsort are not allowed due to non-deterministic behavior.\n"
    "Please use mergesort for deterministic sorting behavior."
)


# Define custom exception for sorting errors
class SortingError(Exception):
    pass


def stable_sort(arr, kind="mergesort", **kwargs):
    """Stable sort function using mergesort, w/o numba optimization.

    Args:
        arr: numpy array to sort
        kind: sorting algorithm to use (only 'mergesort' is allowed)

    Returns:
        Sorted array using mergesort algorithm

    """
    if kind != "mergesort":
        raise SortingError(UNSTABLE_SORT_MESSAGE)
    return np.sort(arr, kind="mergesort", **kwargs)


@register_jitable
def stable_argsort(arr, kind="mergesort"):
    """Numba-optimized stable argsort function using mergesort.

    Args:
        arr: numpy array to sort
        kind: sorting algorithm to use (only 'mergesort' is allowed)

    Returns:
        Indices that would sort the array using mergesort algorithm

    """
    if kind != "mergesort":
        raise SortingError(UNSTABLE_SORT_MESSAGE)
    return np.argsort(arr, kind="mergesort")
