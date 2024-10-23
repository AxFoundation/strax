import numpy as np
import functools
import warnings

# Store original functions
original_sort = np.sort
original_argsort = np.argsort


class SortingError(Exception):
    """Custom exception for sorting violations."""

    pass


def enforce_mergesort(func_name):
    """Raises warning if not using mergesort."""
    warnings.warn(
        f"Direct use of {func_name} detected.\n"
        "Please use the mergesort wrapper to ensure deterministic behavior.",
        UserWarning,
        stacklevel=3,
    )


def create_sort_wrapper(original_func, func_name):
    """Creates a wrapper that enforces mergesort usage."""

    @functools.wraps(original_func)
    def wrapper(arr, *args, **kwargs):
        # Check if explicit quicksort is requested
        if kwargs.get("kind") == "quicksort" or kwargs.get("kind") == "heapsort":
            raise SortingError(
                "quicksort and heapsort are not allowed due to non-deterministic behavior.\n"
                "Please use mergesort explicitely, "
                "or remove the 'kind' parameter to use mergesort by default."
            )

        # Always use mergesort
        kwargs["kind"] = "mergesort"

        # Log usage of direct sorting functions
        if original_func in {np.sort, np.argsort}:
            enforce_mergesort(func_name)

        return original_func(arr, *args, **kwargs)

    return wrapper


# Create wrappers for all sorting functions
sort_wrapper = create_sort_wrapper(original_sort, "np.sort")
argsort_wrapper = create_sort_wrapper(original_argsort, "np.argsort")

# Export wrapped versions for explicit usage
mergesort = sort_wrapper
mergesort_argsort = argsort_wrapper


def enable_safe_sorting():
    """Patches all NumPy sorting methods to enforce mergesort.

    Returns a function to restore original behavior if needed.

    """
    # Replace all sorting methods
    np.sort = sort_wrapper
    np.argsort = argsort_wrapper

    def restore_original_sorts():
        """Restores original NumPy sorting behavior."""
        np.sort = original_sort
        np.argsort = original_argsort

    return restore_original_sorts


# Enable safe sorting immediately
restore_sorts = enable_safe_sorting()
