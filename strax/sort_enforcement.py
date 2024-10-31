import numpy as np
import numba

# Define error message as a constant
UNSTABLE_SORT_MESSAGE = (
    "quicksort and heapsort are not allowed due to non-deterministic behavior.\n"
    "Please use mergesort for deterministic sorting behavior."
)

@numba.njit(nogil=True, cache=True)
def stable_sort(arr, kind='mergesort'):
    """Numba-optimized stable sort function using mergesort.
    
    Args:
        arr: numpy array to sort
        kind: sorting algorithm to use (only 'mergesort' is allowed)
    
    Returns:
        Sorted array using mergesort algorithm
    """
    if kind != 'mergesort':
        raise ValueError(UNSTABLE_SORT_MESSAGE)
    return np.sort(arr)

@numba.njit(nogil=True, cache=True)
def stable_argsort(arr, kind='mergesort'):
    """Numba-optimized stable argsort function using mergesort.
    
    Args:
        arr: numpy array to sort
        kind: sorting algorithm to use (only 'mergesort' is allowed)
    
    Returns:
        Indices that would sort the array using mergesort algorithm
    """
    if kind != 'mergesort':
        raise ValueError(UNSTABLE_SORT_MESSAGE)
    return np.argsort(arr, kind='mergesort')
