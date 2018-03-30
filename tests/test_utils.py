from .helpers import sorted_intervals, disjoint_sorted_intervals

import numpy as np
import strax

from hypothesis import given


def test_growing_result():
    @strax.growing_result(np.int, chunk_size=2)
    def bla(_result_buffer=None, result_dtype=None):
        buffer = _result_buffer
        offset = 0

        for i in range(5):
            buffer[offset] = i

            offset += 1
            if offset == len(buffer):
                yield offset
                offset = 0
        yield offset

    result = np.array([0, 1, 2, 3, 4], dtype=np.int)
    np.testing.assert_equal(bla(), result)
    # TODO: re-enable chunk size spec?
    # np.testing.assert_equal(bla(chunk_size=1), result)
    # np.testing.assert_equal(bla(chunk_size=7), result)
    should_get = result.astype(np.float)
    got = bla(result_dtype=np.float)
    np.testing.assert_equal(got, should_get)
    assert got.dtype == should_get.dtype


@given(sorted_intervals, disjoint_sorted_intervals)
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
def test_split_by_containment(things, containers):
    result = strax.split_by_containment(things, containers)

    assert len(result) == len(containers)

    for container_i, things_in in enumerate(result):
        for t in things:
            assert ((t in things_in)
                    == _is_contained(t, containers[container_i]))


def _is_contained(_thing, _container):
    # Assumes dt = 1
    return _container['time'] \
           <= _thing['time'] \
           <= _thing['time'] + _thing['length'] \
           <= _container['time'] + _container['length']
