import numpy as np
import pytest
from . import helpers   # Mocks numba    # noqa
from strax.chunk_arrays import ChunkPacer, fixed_size_chunks, same_length
from strax.chunk_arrays import fixed_length_chunks
from strax.chunk_arrays import same_stop, sync_iters


@pytest.fixture
def source():
    def f():
        for i in range(10):
            yield np.arange(100, dtype=np.int64) + 100 * i
    return f()


@pytest.fixture
def source_2():
    def f():
        for i in range(100):
            yield np.arange(10, dtype=np.int64) + 10 * i
    return f()


def test_get_next(source):
    p = ChunkPacer(source)
    result = []
    while True:
        try:
            result.append(p.get_n(42))
        except StopIteration:
            break

    assert np.all(np.array([len(x) for x in result[:-1]])
                  == 42)
    assert len(result[-1]) == 1000 % 42
    _check_mangling(result)


def _check_mangling(result, total_length=1000, diff=1):
    r_concat = np.concatenate(result)
    assert len(r_concat) == total_length, "Length mangled"
    assert np.all(np.diff(r_concat) == diff), "Order mangled"


def test_get_until(source):
    p = ChunkPacer(source)
    result = []
    thresholds = [123.5, 321.5, 456.5]
    for t in thresholds:
        result.append(p.get_until(t))

    _check_mangling(result, total_length=457)

    result.append(p.get_until(678))

    assert all([result[i][-1] < thresholds[i]
                for i in range(len(thresholds))])
    assert all([result[i + 1][0] >= thresholds[i]
                for i in range(len(thresholds) - 1)])


def test_fixed_length_chunks(source):
    # TODO: test chunk size < array
    # test chunk size > array
    result = list(fixed_length_chunks(source, int(1e9)))
    _check_mangling(result)


def test_fixed_size_chunks(source):
    # test chunk size < array
    result = list(fixed_size_chunks(source, 42 * 8))
    assert np.all(np.array([len(x) for x in result[:-1]])
                  == 42)
    assert len(result[-1]) == 1000 % 42
    _check_mangling(result)


def test_fixed_size_chunks_oversized(source):
    # test chunk size > array
    result = list(fixed_size_chunks(source, int(1e9)))
    _check_mangling(result)


def test_same_length(source, source_2):
    result = list(same_length(source, source_2))
    assert all(len(x[0]) == len(x[1]) for x in result)
    _check_mangling([x[0] for x in result])
    _check_mangling([x[1] for x in result])


@pytest.fixture
def source_skipper():
    def f():
        for i in range(100):
            yield np.arange(0, 10, 2) + 10 * i
    return f()


def test_same_stop(source, source_skipper):
    result = list(same_stop(source, source_skipper))

    _do_sync_check([x[0] for x in result],
                   [x[1] for x in result])


def _do_sync_check(r1, r2):
    _check_mangling(r1)
    _check_mangling(r2, total_length=500, diff=2)
    for i in range(len(r1)):
        assert r2[i][-1] <= r1[i][-1]
        if i != len(r1) - 1:
            assert r2[i + 1][-1] > r1[i][-1]


def test_sync_iters(source, source_skipper):
    synced = sync_iters(same_stop, dict(s1=source, s2=source_skipper))
    assert len(synced) == 2
    assert 's1' in synced and 's2' in synced

    _do_sync_check(list(synced['s1']),
                   list(synced['s2']))
