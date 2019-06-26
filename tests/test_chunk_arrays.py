import itertools
from .helpers import *


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


class CorrectlyExhaustedEmptySource(Exception):
    pass


def _s_empties(crash=True):
    items_yielded = 0
    for i in itertools.count():
        if i % 2:
            yield np.ones(0, dtype=np.int64)
        else:
            yield np.ones(1, dtype=np.int64) * items_yielded
            items_yielded += 1
        if items_yielded == 1000:
            if crash:
                raise CorrectlyExhaustedEmptySource
            return


@pytest.fixture
def source_some_empty_crasher():
    # Source that yields a mix of empty and non-empty arrays
    # Tests should check custom exception at end gets raised
    return _s_empties(crash=True)


@pytest.fixture
def source_some_empty():
    return _s_empties(crash=False)


def test_get_next(source):
    p = strax.ChunkPacer(source)

    # Getting nothing results in empty array
    r = p.get_n(0)
    assert len(r) == 0
    assert r.dtype == np.int64

    result = []
    while not p.exhausted:
        result.append(p.get_n(42))

    assert np.all(np.array([len(x) for x in result[:-1]])
                  == 42)
    assert len(result[-1]) == 1000 % 42
    _check_mangling(result)


def _check_mangling(result, total_length=1000, diff=1):
    r_concat = np.concatenate(result)
    assert len(r_concat) == total_length, "Length mangled"
    assert np.all(np.diff(r_concat) == diff), "Order mangled"


def test_get_until(source):
    p = strax.ChunkPacer(source)
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

    result.append(p.get_until(5000))
    _check_mangling(result)

    # Regression test
    p.get_until(5000)
    assert p.exhausted


def test_get_some_emty(source_some_empty_crasher):
    p = strax.ChunkPacer(source_some_empty_crasher)
    p.get_n(1000)
    with pytest.raises(CorrectlyExhaustedEmptySource):
        p.get_n(1)


def test_fixed_length_chunks(source):
    # TODO: test chunk size < array
    # test chunk size > array
    result = list(strax.fixed_length_chunks(source, int(1e9)))
    _check_mangling(result)


def test_fixed_size_chunks(source):
    # test chunk size < array
    result = list(strax.fixed_size_chunks(source, 42 * 8))
    assert np.all(np.array([len(x) for x in result[:-1]])
                  == 42)
    assert len(result[-1]) == 1000 % 42
    _check_mangling(result)


def test_fixed_size_chunks_oversized(source):
    # test chunk size > array
    result = list(strax.fixed_size_chunks(source, int(1e9)))
    _check_mangling(result)


def test_same_length(source, source_2):
    result = list(strax.same_length(source, source_2))
    assert all(len(x[0]) == len(x[1]) for x in result)
    _check_mangling([x[0] for x in result])
    _check_mangling([x[1] for x in result])


@pytest.fixture
def source_skipper():
    # Source that only returns even numbers from 0-1000
    def f():
        for i in range(100):
            yield np.arange(0, 10, 2) + 10 * i
    return f()


def test_same_stop(source, source_skipper):
    result = list(strax.same_stop(source, source_skipper))

    _do_sync_check([x[0] for x in result],
                   [x[1] for x in result])


def test_same_stop_some_empty(source_some_empty, source_skipper):
    result = list(strax.same_stop(source_some_empty, source_skipper))

    _do_sync_check([x[0] for x in result],
                   [x[1] for x in result])


def _do_sync_check(r1, r2):
    assert len(r1) == len(r2)
    # Checks order and length are OK
    _check_mangling(r1)
    _check_mangling(r2, total_length=500, diff=2)

    seen_r1 = 0
    seen_r2 = 0
    for i in range(len(r1)):
        # If pacemaker is empty, other r is also
        if not len(r1[i]):
            assert not len(r2[i])
            continue

        # Second r does not lag behind first
        if len(r2[i]):
            assert min(r2[i]) >= seen_r1
            seen_r2 = max(r2[i])

        seen_r1 = max(r1[i])

        # Second r does not outpace first
        assert seen_r1 >= seen_r2


def test_sync_iters(source, source_skipper):
    synced = strax.sync_iters(strax.same_stop,
                              dict(s1=source, s2=source_skipper))
    assert len(synced) == 2
    assert 's1' in synced and 's2' in synced

    _do_sync_check(list(synced['s1']),
                   list(synced['s2']))
