from hypothesis import given, example
from .helpers import sorted_intervals, disjoint_sorted_intervals
from .helpers import several_fake_records

import numpy as np
import strax


@given(sorted_intervals, disjoint_sorted_intervals)
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
        assert np.diff(np.concatenate(result)['time']) >= 0, "Sorting broken"


def _is_contained(_thing, _container):
    # Assumes dt = 1
    return _container['time'] \
           <= _thing['time'] \
           <= _thing['time'] + _thing['length'] \
           <= _container['time'] + _container['length']


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
