import unittest
import numpy as np
import warnings
from strax.sort_enforcement import SortingError, mergesort, mergesort_argsort, restore_sorts


class TestSortEnforcement(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        # Store expected sorted array and indices for comparison
        self.expected_sorted = np.array([1, 1, 2, 3, 4, 5, 6, 9])
        self.expected_argsort = np.array([1, 3, 6, 0, 2, 4, 7, 5])

    def test_explicit_mergesort(self):
        """Test explicit mergesort function (should not warn)"""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            sorted_arr = mergesort(self.arr)
            np.testing.assert_array_equal(sorted_arr, self.expected_sorted)

    def test_explicit_mergesort_argsort(self):
        """Test explicit mergesort_argsort function (should not warn)"""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            sorted_indices = mergesort_argsort(self.arr)
            np.testing.assert_array_equal(sorted_indices, self.expected_argsort)

    def test_quicksort_rejection(self):
        """Test that quicksort and heapsort raise errors for both sort and argsort."""
        # Test np.sort
        with self.assertRaises(SortingError):
            np.sort(self.arr, kind="quicksort")
        with self.assertRaises(SortingError):
            np.sort(self.arr, kind="heapsort")

        # Test np.argsort
        with self.assertRaises(SortingError):
            np.argsort(self.arr, kind="quicksort")
        with self.assertRaises(SortingError):
            np.argsort(self.arr, kind="heapsort")

    def test_sort_stability(self):
        """Test that sorting is stable (mergesort property)"""
        # Create array with duplicate values
        arr = np.array(
            [(1, "a"), (2, "b"), (1, "c"), (2, "d")], dtype=[("num", int), ("letter", "U1")]
        )
        sorted_arr = np.sort(arr, order="num")
        # Check that relative order of equal elements is preserved
        self.assertEqual(sorted_arr[0]["letter"], "a")
        self.assertEqual(sorted_arr[1]["letter"], "c")
        self.assertEqual(sorted_arr[2]["letter"], "b")
        self.assertEqual(sorted_arr[3]["letter"], "d")

    def test_restore_functionality(self):
        """Test that restore_sorts function works correctly."""
        # First verify mergesort is enforced
        with self.assertRaises(SortingError):
            np.sort(self.arr, kind="quicksort")

        # Restore original behavior
        restore_sorts()

        # Now quicksort should work without error
        try:
            np.sort(self.arr, kind="quicksort")
        except SortingError:
            self.fail("quicksort raised SortingError after restore")


if __name__ == "__main__":
    unittest.main(verbosity=2)
