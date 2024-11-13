import unittest
import numpy as np
import warnings
from hypothesis import given, strategies, settings, assume
from hypothesis.extra.numpy import arrays, integer_dtypes, unicode_string_dtypes
from strax.sort_enforcement import SortingError, stable_sort, stable_argsort


class TestSortEnforcement(unittest.TestCase):
    @given(arrays(integer_dtypes(), strategies.integers(1, 100)))
    def test_explicit_stable_sort(self, arr):
        """Test explicit stable_sort function with generated arrays."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            sorted_arr = stable_sort(arr)
            np.testing.assert_array_equal(sorted_arr, np.sort(arr, kind="mergesort"))
            # Verify the array is actually sorted
            self.assertTrue(np.all(sorted_arr[:-1] <= sorted_arr[1:]))

    @given(arrays(integer_dtypes(), strategies.integers(1, 100)))
    def test_explicit_stable_argsort(self, arr):
        """Test explicit stable_argsort function with generated arrays."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            sorted_indices = stable_argsort(arr)
            np.testing.assert_array_equal(sorted_indices, np.argsort(arr, kind="mergesort"))
            # Verify the indices actually sort the array
            sorted_arr = arr[sorted_indices]
            self.assertTrue(np.all(sorted_arr[:-1] <= sorted_arr[1:]))

    @given(
        arrays(integer_dtypes(), strategies.integers(1, 100)),
        strategies.sampled_from(["quicksort", "heapsort"]),
    )
    def test_wrapped_quicksort_rejection(self, arr, sort_kind):
        """Test that quicksort and heapsort raise errors in wrapped functions."""
        with self.assertRaises(SortingError):
            stable_sort(arr, kind=sort_kind)
        with self.assertRaises(SortingError):
            stable_argsort(arr, kind=sort_kind)

    @given(arrays(integer_dtypes(), strategies.integers(1, 100)))
    def test_original_numpy_unaffected(self, arr):
        """Test that original numpy sort functions still work with quicksort."""
        try:
            quicksort_result = np.sort(arr, kind="quicksort")
            self.assertTrue(np.all(quicksort_result[:-1] <= quicksort_result[1:]))

            quicksort_indices = np.argsort(arr, kind="quicksort")
            sorted_arr = arr[quicksort_indices]
            self.assertTrue(np.all(sorted_arr[:-1] <= sorted_arr[1:]))
        except Exception as e:
            self.fail(f"numpy sort with quicksort raised an unexpected exception: {e}")

    @given(
        strategies.lists(
            strategies.tuples(
                strategies.integers(1, 10),  # num field
                strategies.text(min_size=1, max_size=1),  # letter field
            ),
            min_size=1,
            max_size=100,
        )
    )
    def test_sort_stability(self, data):
        """Test that wrapped sorting is stable using generated structured arrays."""
        # Convert list of tuples to structured array
        arr = np.array(data, dtype=[("num", int), ("letter", "U1")])

        # First sort by letter to establish initial order
        arr_by_letter = stable_sort(arr, order="letter")
        # Then sort by number - if sort is stable, items with same number
        # should maintain their relative order from the letter sort
        final_sort = stable_sort(arr_by_letter, order="num")

        # Verify sorting works correctly
        for i in range(len(final_sort) - 1):
            # Check primary sort key (number)
            self.assertTrue(
                final_sort[i]["num"] <= final_sort[i + 1]["num"],
                f"Primary sort failed: {final_sort[i]} should come before {final_sort[i + 1]}",
            )

            # If numbers are equal, check that letter order is preserved
            if final_sort[i]["num"] == final_sort[i + 1]["num"]:
                self.assertTrue(
                    final_sort[i]["letter"] <= final_sort[i + 1]["letter"],
                    f"Stability violated: for equal numbers {final_sort[i]['num']}, "
                    f"letter {final_sort[i]['letter']} should come before or equal to {final_sort[i + 1]['letter']}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
