from hypothesis import given, example, settings
from hypothesis.strategies import integers
from strax.testutils import sorted_intervals, disjoint_sorted_intervals
from strax.testutils import several_fake_records

import numpy as np
import strax


@given(sorted_intervals, disjoint_sorted_intervals)
@settings(deadline=None)
# Tricky example: uncontained interval precedes contained interval
# (this did not produce an issue, but good to show this is handled)
@example(things=np.array([(0, 1, 0, 1),
                          (0, 1, 1, 5),
                          (0, 1, 2, 1)],
                         dtype=strax.interval_dtype),
         containers=np.array([(0, 1, 0, 4)],
                             dtype=strax.interval_dtype))
def test_fully_contained_in(things, containers):
    result = strax.fully_contained_in(things, containers)

    assert len(result) == len(things)
    if len(result):
        assert result.max() < len(containers)

    for i, thing in enumerate(things):
        if result[i] == -1:
            # Check for false negative
            for c in containers:
                assert not _is_contained(thing, c)
        else:
            # Check for false positives
            assert _is_contained(thing, containers[result[i]])


@settings(deadline=None)
@given(sorted_intervals, disjoint_sorted_intervals)
# Specific example to trigger issue #37
@example(
    things=np.array([(0, 1, 2, 1)],
                    dtype=strax.interval_dtype),
    containers=np.array([(0, 1, 0, 1), (0, 1, 2, 1)],
                        dtype=strax.interval_dtype))
def test_split_by_containment(things, containers):
    result = strax.split_by_containment(things, containers)

    assert len(result) == len(containers)

    for container_i, things_in in enumerate(result):
        for t in things:
            assert ((t in things_in)
                    == _is_contained(t, containers[container_i]))

    if len(result) and len(np.concatenate(result)) > 1:
        assert np.diff(np.concatenate(result)['time']).min() >= 0, "Sorting bug"


def _is_contained(_thing, _container):
    # Assumes dt = 1
    return _container['time'] \
           <= _thing['time'] \
           <= _thing['time'] + _thing['length'] \
           <= _container['time'] + _container['length']


@settings(deadline=None)
@given(several_fake_records)
def test_from_break(records):
    window = 5

    def has_break(x):
        if len(x) < 2:
            return False
        return np.diff(x['time']).max() > window

    try:
        left = strax.from_break(records, safe_break=window,
                                left=True, tolerant=False)
        right = strax.from_break(records, safe_break=window,
                                 left=False, tolerant=False)
    except strax.NoBreakFound:
        assert not has_break(records)

    else:
        assert len(left) + len(right) == len(records)
        if len(records) > 0:
            np.testing.assert_equal(np.concatenate([left, right]),
                                    records)
        if len(left) and len(right):
            assert left[-1]['time'] <= right[0]['time'] - window
        assert not has_break(right)


@settings(deadline=None)
@given(integers(0, 100), integers(0, 100), integers(0, 100), integers(0, 100))
def test_overlap_indices(a1, n_a, b1, n_b):
    a2 = a1 + n_a
    b2 = b1 + n_b

    (a_start, a_end), (b_start, b_end) = strax.overlap_indices(a1, n_a, b1, n_b)
    assert a_end - a_start == b_end - b_start, "Overlap must be equal length"
    assert a_end >= a_start, "Overlap must be nonnegative"

    if n_a == 0 or n_b == 0:
        assert a_start == a_end == b_start == b_end == 0
        return

    a_filled = np.arange(a1, a2)
    b_filled = np.arange(b1, b2)
    true_overlap = np.intersect1d(a_filled, b_filled)
    if not len(true_overlap):
        assert a_start == a_end == b_start == b_end == 0
        return

    true_a_inds = np.searchsorted(a_filled, true_overlap)
    true_b_inds = np.searchsorted(b_filled, true_overlap)

    found = (a_start, a_end), (b_start, b_end)
    expected = (
        (true_a_inds[0], true_a_inds[-1] + 1),
        (true_b_inds[0], true_b_inds[-1] + 1))
    assert found == expected
