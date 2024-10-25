import numpy as np
import functools
import warnings
from copy import deepcopy


class SortingError(Exception):
    """Custom exception for sorting violations."""

    pass


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
        # Check if explicit quicksort is requested
        if kwargs.get("kind") == "quicksort" or kwargs.get("kind") == "heapsort":
            raise SortingError(
                "quicksort and heapsort are not allowed due to non-deterministic behavior.\n"
                "Please set kind to be mergesort explicitly, "
                "or remove the 'kind' parameter to use mergesort by default."
            )

        # Always use mergesort
        kwargs["kind"] = "mergesort"
        return original_func(arr, *args, **kwargs)

    return wrapper


# Create wrapped versions of sort functions
stablesort = create_sort_wrapper(np.sort)
stableargsort = create_sort_wrapper(np.argsort)
