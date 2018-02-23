import numpy as np
import strax

from hypothesis import given
from .helpers import sorted_intervals, disjoint_sorted_intervals


def test_growing_result():
    @strax.growing_result(np.int, chunk_size=2)
    def bla(buffer):
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
    np.testing.assert_equal(bla(chunk_size=1), result)
    np.testing.assert_equal(bla(chunk_size=7), result)
    np.testing.assert_equal(bla(dtype=np.float), result.astype(np.float))


@given(sorted_intervals, disjoint_sorted_intervals)
def test_fully_contained_in(things, containers):
    result = strax.fully_contained_in(things, containers)

    assert len(result) == len(things)
    if len(result):
        assert result.max() < len(containers)

    def is_contained(_thing, _container):
        return _container['time'] \
               <= _thing['time'] \
               <= _thing['time'] + _thing['length'] \
               <= _container['time'] + _container['length']

    for i, thing in enumerate(things):
        if result[i] == -1:
            # Check for false negative
            for c in containers:
                assert not is_contained(thing, c)
        else:
            # Check for false positives
            assert is_contained(thing, containers[result[i]])
