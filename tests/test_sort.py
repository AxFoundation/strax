import unittest
import numpy as np
import warnings
from strax.sort_enforcement import SortingError, stable_sort, stable_argsort


class TestSortEnforcement(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        # Store expected sorted array and indices for comparison
        self.expected_sorted = np.array([1, 1, 2, 3, 4, 5, 6, 9])
        self.expected_argsort = np.array([1, 3, 6, 0, 2, 4, 7, 5])

    def test_explicit_stable_sort(self):
        """Test explicit stable_sort function (should not warn)"""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            sorted_arr = stable_sort(self.arr)
            np.testing.assert_array_equal(sorted_arr, self.expected_sorted)

    def test_explicit_stable_sort_argsort(self):
        """Test explicit stable_sort_argsort function (should not warn)"""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            sorted_indices = stable_argsort(self.arr)
            np.testing.assert_array_equal(sorted_indices, self.expected_argsort)

    def test_wrapped_quicksort_rejection(self):
        """Test that quicksort and heapsort raise errors in wrapped functions."""
        # Test stable_sort wrapper
        with self.assertRaises(SortingError):
            stable_sort(self.arr, kind="quicksort")
        with self.assertRaises(SortingError):
            stable_sort(self.arr, kind="heapsort")

        # Test stable_sort_argsort wrapper
        with self.assertRaises(SortingError):
            stable_argsort(self.arr, kind="quicksort")
        with self.assertRaises(SortingError):
            stable_argsort(self.arr, kind="heapsort")

    def test_original_numpy_unaffected(self):
        """Test that original numpy sort functions still work with quicksort."""
        # Test np.sort with quicksort
        try:
            quicksort_result = np.sort(self.arr, kind="quicksort")
            np.testing.assert_array_equal(quicksort_result, self.expected_sorted)
        except Exception as e:
            self.fail(f"np.sort with quicksort raised an unexpected exception: {e}")

        # Test np.argsort with quicksort
        try:
            quicksort_indices = np.argsort(self.arr, kind="quicksort")
            # Note: We don't check exact equality because quicksort might give different 
            # but valid ordering for equal elements
            self.assertTrue(np.all(self.arr[quicksort_indices] == self.expected_sorted))
        except Exception as e:
            self.fail(f"np.argsort with quicksort raised an unexpected exception: {e}")

    def test_sort_stability(self):
        """Test that wrapped sorting is stable (stable_sort property)"""
        # Create array with duplicate values
        arr = np.array(
            [(1, "a"), (2, "b"), (1, "c"), (2, "d")], 
            dtype=[("num", int), ("letter", "U1")]
        )
        sorted_arr = stable_sort(arr, order="num")
        # Check that relative order of equal elements is preserved
        self.assertEqual(sorted_arr[0]["letter"], "a")
        self.assertEqual(sorted_arr[1]["letter"], "c")
        self.assertEqual(sorted_arr[2]["letter"], "b")
        self.assertEqual(sorted_arr[3]["letter"], "d")


if __name__ == "__main__":
    unittest.main(verbosity=2)