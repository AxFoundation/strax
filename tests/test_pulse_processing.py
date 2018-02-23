import numpy as np
import strax


def test_find_hits():
    """Tests the hitfinder with simple example pulses"""
    for w, should_find_intervals in [
            ([], []),
            ([1], [(0, 1)]),
            ([1, 0], [(0, 1)]),
            ([1, 0, 1], [(0, 1), (2, 3)]),
            ([1, 0, 1, 0], [(0, 1), (2, 3)]),
            ([1, 0, 1, 0, 1], [(0, 1), (2, 3), (4, 5)]),
            ([0, 1, 2, 0, 4, -1, 60, 700, -4], [(1, 3), (4, 5), (6, 8)]),
            ([1, 1, 2, 0, 4, -1, 60, 700, -4], [(0, 3), (4, 5), (6, 8)]),
            ([1, 0, 2, 3, 4, -1, 60, 700, -4], [(0, 1), (2, 5), (6, 8)]),
            ([1, 0, 2, 3, 4, -1, 60, 700, 800], [(0, 1), (2, 5), (6, 9)]),
            ([0, 0, 2, 3, 4, -1, 60, 700, 800], [(2, 5), (6, 9)])]:

        records = np.zeros(1, strax.record_dtype(9))
        records[0]['data'][:len(w)] = w
        records['dt'] = 1
        records['length'] = 9

        hits = strax.find_hits(records, threshold=0)

        np.testing.assert_equal(hits['time'], hits['left'])
        # NB: exclusive right bound, no + 1 here
        np.testing.assert_equal(hits['length'],
                                hits['right'] - hits['left'])

        results = list(zip(hits['left'], hits['right']))
        assert len(results) == len(should_find_intervals)
        assert results == should_find_intervals
