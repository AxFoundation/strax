import numpy as np
import functools
import warnings
import numba

class SortingError(Exception):
    """Custom exception for sorting violations."""
    pass

# Common error message for unstable sort methods
UNSTABLE_SORT_MESSAGE = (
    "quicksort and heapsort are not allowed due to non-deterministic behavior.\n"
    "Please set kind to be mergesort explicitly, "
    "or remove the 'kind' parameter to use mergesort by default."
)

def enforce_stablesort_warning():
    """Issues warning about using stablesort wrapper."""
    warnings.warn(
        "Consider using stablesort or stableargsort to ensure deterministic behavior.",
        UserWarning,
        stacklevel=3,
    )

def create_sort_wrapper(original_func):
    """Creates a wrapper that enforces stablesort usage."""
    @functools.wraps(original_func)
    def wrapper(arr, *args, **kwargs):
        # Check for explicitly disallowed sorting methods
        if kwargs.get("kind") in ("quicksort", "heapsort"):
            raise SortingError(UNSTABLE_SORT_MESSAGE)

        # Enforce mergesort
        kwargs["kind"] = "mergesort"
        return original_func(arr, *args, **kwargs)

    return wrapper

def create_numba_stablesort(sort_func):
    """Creates a Numba-optimized stable sort function."""
    @numba.njit(nogil=True, cache=True)
    def _stable_sort(arr, kind='mergesort'):
        if kind in ('quicksort', 'heapsort'):
            raise ValueError(UNSTABLE_SORT_MESSAGE)
        return sort_func(arr, kind='mergesort')
    return _stable_sort

# Create Numba-optimized versions
numba_stablesort = create_numba_stablesort(np.sort)
numba_stableargsort = create_numba_stablesort(np.argsort)

# Create wrapped versions of regular numpy sort functions
stablesort = create_sort_wrapper(np.sort)
stableargsort = create_sort_wrapper(np.argsort)