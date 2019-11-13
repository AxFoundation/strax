from strax import testutils
from hypothesis import given
import strax


@given(testutils.sorted_bounds())
def test_sorted_bounds(bs):
    assert is_sorted(bs)


@given(testutils.sorted_bounds(disjoint=True))
def test_disjoint_bounds(bs):
    assert is_sorted(bs)
    assert is_disjoint(bs)


@given(testutils.disjoint_sorted_intervals)
def test_dsi(intvs):
    bs = list(zip(intvs['time'].tolist(), strax.endtime(intvs).tolist()))
    assert is_sorted(bs)
    assert is_disjoint(bs)


def is_sorted(bs):
    return bs == sorted(bs)


def is_disjoint(bs):
    return all([bs[i][1] <= bs[i + 1][0]
                for i in range(len(bs) - 1)])
