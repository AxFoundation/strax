from hypothesis.extra import numpy as hyp_numpy
import hypothesis.strategies

import unittest
import strax.testutils

import numpy as np
import strax


@hypothesis.given(strax.testutils.sorted_intervals, strax.testutils.disjoint_sorted_intervals)
@hypothesis.settings(deadline=None)
# Tricky example: uncontained interval precedes contained interval
# (this did not produce an issue, but good to show this is handled)
@hypothesis.example(
    things=np.array([(0, 1, 1, 0), (1, 5, 1, 0), (2, 1, 1, 0)], dtype=strax.interval_dtype),
    containers=np.array([(0, 4, 1, 0)], dtype=strax.interval_dtype),
)
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


@hypothesis.given(
    strax.testutils.sorted_intervals,
    strax.testutils.disjoint_sorted_intervals,
    hypothesis.strategies.integers(-2, 2),
)
@hypothesis.settings(deadline=None)
@hypothesis.example(
    things=np.array([(0, 1, 1, 0), (1, 5, 1, 0), (2, 1, 1, 0)], dtype=strax.interval_dtype),
    containers=np.array([(0, 4, 1, 0)], dtype=strax.interval_dtype),
    window=0,
)
def test_touching_windows(things, containers, window):
    result = strax.touching_windows(things, containers, window=window)
    assert len(result) == len(containers)
    if len(result):
        assert np.all((0 <= result) & (result <= len(things)))

    for c_i, container in enumerate(containers):
        i_that_touch = np.arange(*result[c_i])
        for t_i, thing in enumerate(things):
            if (
                strax.endtime(thing) <= container["time"] - window
                or thing["time"] >= strax.endtime(container) + window
            ):
                assert t_i not in i_that_touch
            else:
                assert t_i in i_that_touch


@hypothesis.settings(deadline=None, max_examples=1000)
@hypothesis.given(strax.testutils.sorted_intervals, strax.testutils.disjoint_sorted_intervals)
# Specific example to trigger issue #37
@hypothesis.example(
    things=np.array([(2, 1, 1, 0)], dtype=strax.interval_dtype),
    containers=np.array([(0, 1, 1, 0), (2, 1, 1, 0)], dtype=strax.interval_dtype),
)
def test_split_by_containment(things, containers):
    result = strax.split_by_containment(things, containers)

    assert len(result) == len(containers)

    for container_i, things_in in enumerate(result):
        for t in things:
            assert (t in things_in) == _is_contained(t, containers[container_i])

    if len(result) and len(np.concatenate(result)) > 1:
        assert np.diff(np.concatenate(result)["time"]).min() >= 0, "Sorting bug"


def _is_contained(_thing, _container):
    # Assumes dt = 1
    return (
        _container["time"]
        <= _thing["time"]
        <= _thing["time"] + _thing["length"]
        <= _container["time"] + _container["length"]
    )


@hypothesis.given(
    full_container_ids=hyp_numpy.arrays(
        np.int8,
        hypothesis.strategies.integers(0, 10),
        elements=hypothesis.strategies.integers(0, 15),
        unique=True,
    )
)
@hypothesis.settings(deadline=None)
def test_get_empty_container_ids(full_container_ids):
    """Helper function to test if strax.processing.general._get_empty_container_ids behaves the same
    as np.setdiff1d. Both functions should compute the a set diff of the two arrays. E.g. in this
    case an array with unique numbers between 0 and 15 which are not in full_container_ids.

    :param full_container_ids: Array which mimics the ids of full containers. the array has a size
        between 0 and 10 and is filled with integer values between 0 and 15.
    :return:

    """
    full_container_ids = np.sort(full_container_ids)

    if len(full_container_ids):
        n_containers = np.max(full_container_ids)
    else:
        n_containers = 0

    empty_ids = strax.processing.general._get_empty_container_ids(n_containers, full_container_ids)

    empty_ids_np = np.setdiff1d(np.arange(n_containers), full_container_ids)

    mes = (
        "_get_empty_containers_ids did not match the behavior or np.setdiff1d.\n"
        + f"_get_empty_containers_ids: {empty_ids}\n"
        + f"np.setdiff1d: {empty_ids_np}"
    )

    assert np.all(empty_ids == empty_ids_np), mes


@hypothesis.given(
    things=hyp_numpy.arrays(
        np.int8,
        hypothesis.strategies.integers(0, 10),
        elements=hypothesis.strategies.integers(0, 10),
        unique=False,
    ),
    split_indices=hyp_numpy.arrays(
        np.int8,
        hypothesis.strategies.integers(0, 10),
        elements=hypothesis.strategies.integers(0, 10),
        unique=True,
    ),
)
@hypothesis.settings(deadline=None)
def test_split(things, split_indices):
    """Test to check if strax.processing.general._split shows the same behavior as np.split for the
    previous split_by_containment function.

    :param things: things to be split. Hypothesis will create here an array of a length between 0
        and 10. Each element in this array can also range between 0 and 10.
    :param split_indices: Indices at which things should be split.

    """
    split_indices = np.sort(split_indices)

    split_things = strax.processing.general._split(things, split_indices)
    split_things_np = np.split(things, split_indices)

    for ci, (s, snp) in enumerate(zip(split_things, split_things_np)):
        # Loop over splitted objects and check if they are the same.
        mes = f"Not all splitted things are the same for split {ci}!"
        assert np.all(s == snp), mes


@hypothesis.settings(deadline=None)
@hypothesis.given(
    strax.testutils.several_fake_records,
    hypothesis.strategies.integers(0, 50),
    hypothesis.strategies.booleans(),
)
def test_split_array(data, t, allow_early_split):
    print(
        f"\nCalled with {np.transpose([data['time'], strax.endtime(data)]).tolist()}, "
        f"{t}, {allow_early_split}"
    )

    try:
        data1, data2, tsplit = strax.split_array(data, t, allow_early_split=allow_early_split)

    except strax.CannotSplit:
        assert not allow_early_split
        # There must be data straddling t
        for d in data:
            if d["time"] < t < strax.endtime(d):
                break
        else:
            raise ValueError("threw CannotSplit needlessly")

    else:
        if allow_early_split:
            assert tsplit <= t
            t = tsplit

        assert len(data1) + len(data2) == len(data)
        assert np.all(strax.endtime(data1) <= t)
        assert np.all(data2["time"] >= t)


@hypothesis.settings(deadline=None)
@hypothesis.given(strax.testutils.several_fake_records)
def test_from_break(records):
    if not len(records):
        return

    window = 5

    def has_break(x):
        if len(x) < 2:
            return False
        for i in range(1, len(x)):
            if strax.endtime(x[:i]).max() + window <= x[i]["time"]:
                return True
        return False

    try:
        left, t_break_l = strax.from_break(records, safe_break=window, left=True, tolerant=False)
        right, t_break_r = strax.from_break(records, safe_break=window, left=False, tolerant=False)
    except strax.NoBreakFound:
        assert not has_break(records)

    else:
        assert t_break_l == t_break_r, "Inconsistent break time"
        t_break = t_break_l

        assert has_break(records), f"Found nonexistent break at {t_break}"

        assert len(left) + len(right) == len(records), "Data loss"

        if len(records) > 0:
            np.testing.assert_equal(np.concatenate([left, right]), records)
        if len(right) and len(left):
            assert t_break == right[0]["time"]
            assert strax.endtime(left).max() <= right[0]["time"] - window
        assert not has_break(right)


@hypothesis.settings(deadline=None)
@hypothesis.given(
    hypothesis.strategies.integers(0, 100),
    hypothesis.strategies.integers(0, 100),
    hypothesis.strategies.integers(0, 100),
    hypothesis.strategies.integers(0, 100),
)
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
    expected = ((true_a_inds[0], true_a_inds[-1] + 1), (true_b_inds[0], true_b_inds[-1] + 1))
    assert found == expected


@hypothesis.settings(deadline=None)
@hypothesis.given(strax.testutils.several_fake_records)
def test_raw_to_records(r):
    buffer = np.zeros(len(r), r.dtype)
    strax.copy_to_buffer(r, buffer, "_test_r_to_buffer")
    if len(r):
        assert np.all(buffer == r)


def test_dtype_change_copy_to_buffer():
    """Tests if dtype change does not let copy to buffer fail, even if copy function name stays the
    same."""
    peaks = np.ones(1, dtype=strax.peak_dtype(digitize_top=True))
    buffer = np.zeros(1, dtype=strax.peak_dtype(digitize_top=True))

    strax.copy_to_buffer(peaks, buffer, "_test_copy_to_buffer")
    assert np.all(peaks == buffer)

    peaks = np.ones(1, dtype=strax.peak_dtype(digitize_top=False))
    buffer = np.zeros(1, dtype=strax.peak_dtype(digitize_top=False))

    strax.copy_to_buffer(peaks, buffer, "_test_copy_to_buffer")
    assert np.all(peaks == buffer)


@hypothesis.given(
    hyp_numpy.arrays(np.int64, 10**2, elements=hypothesis.strategies.integers(10**9, 5 * 10**9)),
    hyp_numpy.arrays(np.int16, 10**2, elements=hypothesis.strategies.integers(-10, 10**3)),
)
@hypothesis.settings(deadline=None)
def test_sort_by_time(time, channel):
    n_events = len(time)
    dummy_array = np.zeros(n_events, strax.time_fields)
    dummy_array2 = np.zeros(
        n_events, strax.time_fields + [(("dummy_channel_number", "channel"), np.int16)]
    )
    dummy_array["time"] = time
    dummy_array2["time"] = time

    res1 = strax.sort_by_time(dummy_array)
    res2 = np.sort(dummy_array, order="time")
    assert np.all(res1 == res2)

    res1 = strax.sort_by_time(dummy_array2)
    res2 = np.sort(dummy_array2, order="time")
    assert np.all(res1 == res2)

    # Test again with random channels
    dummy_array3 = dummy_array2.copy()
    dummy_array3["channel"] = channel
    res1 = strax.sort_by_time(dummy_array3)
    res2 = np.sort(dummy_array3, order=("time", "channel"))
    assert np.all(res1 == res2)

    # Create some large time difference that would cause
    # a bug before https://github.com/AxFoundation/strax/pull/695
    # Do make sure that we can actually fit the time difference in an int.64 (hence the //2 +- 1)
    dummy_array3["time"][0] = np.iinfo(np.int64).min // 2 + 1
    dummy_array3["time"][-1] = np.iinfo(np.int64).max // 2 - 1
    res1 = strax.sort_by_time(dummy_array3)
    res2 = np.sort(dummy_array3, order=("time", "channel"))
    assert np.all(res1 == res2)

    _test_sort_by_time_peaks(time)


def _test_sort_by_time_peaks(time):
    """Explicit test to check if peaks a re sorted correctly."""
    dummy_array = np.zeros(
        len(time), strax.time_fields + [(("dummy_channel_number", "channel"), np.int16)]
    )
    dummy_array["time"] = time
    dummy_array["channel"] = -1

    res1 = strax.sort_by_time(dummy_array)
    res2 = np.sort(dummy_array, order="time")
    assert np.all(res1 == res2)


class Test_abs_time_to_prev_next_interval(unittest.TestCase):
    def test_empty_inputs(self):
        events = np.zeros(1, strax.time_fields)
        vetos = np.zeros(0, strax.time_fields)

        events["time"] = 1
        events["endtime"] = 2
        res_prev, res_next = strax.abs_time_to_prev_next_interval(events, vetos)
        assert res_prev == res_next, f"{res_prev}, {res_next}"
        assert np.all(res_prev == -1)

        events = np.zeros(0, strax.time_fields)
        vetos = np.zeros(1, strax.time_fields)

        vetos["time"] = 1
        vetos["endtime"] = 2
        res_prev, res_next = strax.abs_time_to_prev_next_interval(events, vetos)

        _results_are_empty = (len(res_prev) == 0) and (len(res_next) == 0)
        assert _results_are_empty

        events = np.zeros(0, strax.time_fields)
        vetos = np.zeros(0, strax.time_fields)
        res_prev, res_next = strax.abs_time_to_prev_next_interval(events, vetos)

    def test_with_dt_fields(self):
        pass

    @hypothesis.given(
        things=hyp_numpy.arrays(
            np.int64,
            hypothesis.strategies.integers(1, 10),
            elements=hypothesis.strategies.integers(0, 1000),
            unique=False,
        ),
        intervals=hyp_numpy.arrays(
            np.int64,
            hypothesis.strategies.integers(1, 100),
            elements=hypothesis.strategies.integers(0, 100),
            unique=True,
        ),
    )
    @hypothesis.settings(deadline=None)
    def test_correct_time_delays(self, things, intervals):
        _things, _intervals = self._make_correct_things_and_intervals(things, intervals)
        res_prev, res_next = strax.abs_time_to_prev_next_interval(_things, _intervals)

        # Compare for each event:
        for thing_i, e in enumerate(_things):
            dt_prev = e["time"] - _intervals["endtime"]
            dt_prev[dt_prev < 0] = 99999
            dt_prev = np.min(dt_prev)

            msg = (
                f"Found {res_prev[thing_i]}, but expected {dt_prev}"
                f" for thing number {thing_i} of things: {_things} and intervals: {_intervals}"
            )
            if res_prev[thing_i] != -1:
                assert dt_prev == res_prev[thing_i], msg
            else:
                assert dt_prev == 99999, msg

            dt_next = _intervals["time"] - e["endtime"]
            dt_next[dt_next < 0] = 99999
            dt_next = np.min(dt_next)

            msg = (
                f"Found {res_next[thing_i]}, but expected {dt_next}"
                f" for thing number {thing_i} of things: {_things} and intervals: {_intervals}"
            )
            if res_next[thing_i] != -1:
                assert dt_next == res_next[thing_i], msg
            else:
                assert dt_next == 99999, msg

    def _make_correct_things_and_intervals(self, things, intervals):
        _things = np.zeros(len(things), strax.time_fields)
        _things["time"] = things
        _things["endtime"] = things + 100

        _intervals = np.zeros(len(intervals), strax.time_fields)
        _intervals["time"] = intervals
        _intervals["endtime"] = intervals + 10

        _things = strax.sort_by_time(_things)
        _intervals = strax.sort_by_time(_intervals)

        # Cut down overlapping _things and remove empty intervals to not
        # trigger overlapping error:
        dt = _things["endtime"][:-1] - _things["time"][1:]
        dt[dt < 0] = 0
        _things["endtime"][:-1] = _things["endtime"][:-1] - dt
        _is_not_empty_interval = (_things["endtime"] - _things["time"]) > 0
        _things = _things[_is_not_empty_interval]

        return _things, _intervals
