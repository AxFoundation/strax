import warnings

warnings.simplefilter("always", UserWarning)
# for these fundamental functions, we throw warnings each time they are called

import strax
from strax import stable_sort, stable_argsort
import numba
from numba.typed import List
import numpy as np

export, __all__ = strax.exporter()


@export
def sort_by_time(x):
    """Sorts things.

    Either by time or by time, then channel if both fields are in the given array.

    """
    if len(x) == 0:
        return x

    if "channel" in x.dtype.names:
        min_channel = x["channel"].min()
        channel = x["channel"].copy()
        if min_channel < 0:
            channel -= min_channel
    else:
        channel = np.ones(len(x))

    max_time_difference = (np.iinfo(np.int64).max - 10) / (channel.max() + 1)
    # Subtract 10 to have some extra margin, just in case.
    # Use absolute to account for peaks which are channel -1.
    _time_range_too_large = (x["time"].max() - x["time"].min()) > max_time_difference
    if not _time_range_too_large:
        # Faster sorting:
        x = _sort_by_time_and_channel(x, channel, channel.max() + 1)
    elif "channel" in x.dtype.names:
        x = stable_sort(x, order=("time", "channel"))
    else:
        x = stable_sort(x, order=("time",))
    return x


@numba.njit(nogil=True, cache=True)
def _sort_by_time_and_channel(x, channel, max_channel_plus_one, sort_kind="mergesort"):
    """Assumes you have no more than 10k channels, and records don't span more than 11 days.

    (5-10x) faster than strax.stable_sort(order=...), as strax.stable_sort looks at all fields

    """
    # I couldn't get fast argsort on multiple keys to work in numba
    # So, let's make a single key...
    sort_key = (x["time"] - x["time"].min()) * max_channel_plus_one + channel
    sort_i = stable_argsort(sort_key, kind=sort_kind)
    return x[sort_i]


@export
def endtime(x):
    """Return endtime of intervals x."""
    if "endtime" in x.dtype.fields:
        return x["endtime"]
    else:
        return x["time"] + x["length"] * x["dt"]


# Jitting endtime needs special attention, since inspecting the dtype
# has to happen in the python layer.
# (Used to work through numba.generated_jit, now numba.extending.overload)
@numba.extending.overload(endtime)
def _overload_endtime(x):
    """Return endtime of intervals x."""
    if "endtime" in x.dtype.fields:
        return lambda x: x["endtime"]
    else:
        return lambda x: x["time"] + x["length"] * x["dt"]


@export
@numba.njit(nogil=True, cache=True)
def diff(data):
    """Return time differences between items in data."""
    # we are sure that time is np.int64
    if len(data) == 0:
        return np.zeros(0, dtype=np.int64)
    results = np.zeros(len(data) - 1, dtype=np.int64)
    max_endtime = strax.endtime(data[0])
    for i, (time, endtime) in enumerate(zip(data["time"][1:], strax.endtime(data)[:-1])):
        max_endtime = max(max_endtime, endtime)
        results[i] = time - max_endtime
    return results


@export
@numba.njit(nogil=True, cache=True)
def from_break(x, safe_break, not_before=0, left=True, tolerant=False):
    """Return records on side of a break at least safe_break long If there is no such break, return
    the best break found."""
    if tolerant:
        raise NotImplementedError
    if not len(x):
        raise NotImplementedError("Cannot find breaks in empty data")
    if len(x) == 1:
        raise NoBreakFound()

    break_i = _find_break_i(x, safe_break=safe_break, not_before=not_before)
    break_time = x[break_i]["time"]

    if left:
        return x[:break_i], break_time
    else:
        return x[break_i:], break_time


@export
class NoBreakFound(Exception):
    pass


@export
@numba.njit(nogil=True, cache=True)
def _find_break_i(data, safe_break, not_before):
    """Return first index of element right of the first gap larger than safe_break in data. Assumes
    all x have the same length and are sorted!

    :param tolerant: if no break found, yield an as good as possible break anyway.

    """
    assert len(data) >= 2
    latest_end_seen = max(not_before, strax.endtime(data[0]))
    for i, d in enumerate(data):
        if i == 0:
            continue
        if d["time"] >= latest_end_seen + safe_break:
            return i
        latest_end_seen = max(latest_end_seen, strax.endtime(d))
    raise NoBreakFound


def _fully_contained_in_sanity(things, containers):
    """Since both fully_contained_in and split_by_containment use the same core function
    _fully_contained_in, we check the sanity of the inputs here."""
    try:
        _check_time_is_sorted(things["time"])
    except Exception:
        raise ValueError("time of things should be sorted!")
    try:
        _check_time_is_sorted(containers["time"])
    except Exception:
        raise ValueError("time of containers should be sorted!")
    try:
        _check_objects_are_not_overlapping(containers)
    except Exception:
        warnings.warn(
            "Overlapping of containers detected! "
            "fully_contained_in function will only return "
            "the first container of the thing."
        )
    for objects, names in zip([things, containers], ["things", "containers"]):
        try:
            _check_objects_non_negative_length(objects)
        except Exception:
            raise ValueError(f"{names} should have non-negative length!")


@export
def fully_contained_in(things, containers):
    """Return array of len(things) with index of interval in containers for which things are fully
    contained in a container, or -1 if no such exists.

    We assume all things and containers are sorted by time. If containers are overlapping, the first
    container of the thing is chosen.

    """
    _fully_contained_in_sanity(things, containers)

    return _fully_contained_in(things, containers)


@numba.njit(nogil=True, cache=True)
def _fully_contained_in(things, containers):
    """Core function of fully_contained_in."""
    result = np.ones(len(things), dtype=np.int32) * -1
    a_starts = things["time"]
    b_starts = containers["time"]
    a_ends = strax.endtime(things)
    b_ends = strax.endtime(containers)
    _fc_in(a_starts, b_starts, a_ends, b_ends, result)
    return result


@numba.njit(nogil=True, cache=True)
def _fc_in(a_starts, b_starts, a_ends, b_ends, result):
    b_i = 0
    for a_i in range(len(a_starts)):
        # Skip ahead one or more b's if we're beyond them
        # Note <= in second condition: end is an exclusive bound
        while b_i < len(b_starts) and b_ends[b_i] <= a_starts[a_i]:
            b_i += 1
        if b_i == len(b_starts):
            break

        # Check for containment. We only need to check one b, since bs
        # are nonoverlapping
        if b_starts[b_i] <= a_starts[a_i] and a_ends[a_i] <= b_ends[b_i]:
            result[a_i] = b_i


@export
def split_by_containment(things, containers):
    """Return list of thing-arrays contained in each container. Result is returned as a
    numba.typed.List or list if containers are empty.

    Assumes everything is sorted, and containers are non-overlapping.

    """
    _fully_contained_in_sanity(things, containers)

    if not len(containers):
        # No containers so return empty numba.typed.List
        empty_list = List()
        # Small hack to define its type before returning it:
        empty_list.append(np.zeros(0, dtype=things.dtype))
        empty_list = empty_list[:0]
        return empty_list

    return _split_by_containment(things, containers)


@numba.njit(nogil=True, cache=True)
def _split_by_containment(things, containers):
    # Index of which container each thing belongs to, or -1
    which_container = _fully_contained_in(things, containers)

    # Restrict to things in containers
    mask = which_container != -1
    things = things[mask]
    which_container = which_container[mask]
    if not len(things):
        # Return list of empty things in case things are empty,
        # needed to preserve dtype in LoopPlugins.
        things_split = List()
        for _ in range(len(containers)):
            things_split.append(things[:0])
        return things_split

    # Split things up by container
    split_indices = np.where(np.diff(which_container))[0] + 1
    things_split = _split(things, split_indices)

    # Insert empty arrays for empty containers
    empty_containers = _get_empty_container_ids(len(containers), np.unique(which_container))
    for c_i in empty_containers:
        things_split.insert(c_i, things[:0])

    return things_split


@numba.njit(cache=True, nogil=True)
def _split(things, split_indices):
    """Helper to replace np.split, required since numba numpy.split does not return a typed.List.

    Hence outputs cannot be unified.

    """
    things_split = List()
    if len(split_indices):
        # Found split indices so split things up:
        prev_si = 0
        for si in split_indices:
            things_split.append(things[prev_si:si])
            prev_si = si

        if prev_si < len(things):
            # Append things after last gap if exist
            things_split.append(things[prev_si:])
    else:
        # If there are no split indices, all things are in the same
        # container
        things_split.append(things)
    return things_split


@numba.njit(cache=True, nogil=True)
def _get_empty_container_ids(n_containers, full_container_ids):
    """Helper to replace np.setdiff1d for numbafied split_by_containment."""
    res = np.zeros(n_containers, dtype=np.int64)

    n_empty = 0
    prev_fid = 0
    for fid in full_container_ids:
        # Loop over all container ids with input, ids in between
        # must be empty:
        n = fid - prev_fid
        res[n_empty : n_empty + n] = np.arange(prev_fid, fid, dtype=np.int64)
        prev_fid = fid + 1
        n_empty += n

    if prev_fid < n_containers:
        # Do the rest if there is any:
        n = n_containers - prev_fid
        res[n_empty : n_empty + n] = np.arange(prev_fid, n_containers, dtype=np.int64)
        n_empty += n
    return res[:n_empty]


@export
@numba.njit(nogil=True, cache=True)
def overlap_indices(a1, n_a, b1, n_b):
    """Given interval [a1, a1 + n_a), and [b1, b1 + n_b) of integers, return indices [a_start,
    a_end), [b_start, b_end) of overlapping region."""
    if n_a < 0 or n_b < 0:
        raise ValueError("Negative interval length passed to overlap test")

    if n_a == 0 or n_b == 0:
        return (0, 0), (0, 0)

    # a: p, b: r
    s = a1 - b1

    if s <= -n_a:
        # B is completely right of a
        return (0, 0), (0, 0)

    # Range in b that overlaps with a
    b_start = max(0, s)
    b_end = min(n_b, s + n_a)
    if b_start >= b_end:
        # B is completely left of a
        return (0, 0), (0, 0)

    # Range of a that overlaps with b
    a_start = max(0, -s)
    a_end = min(n_a, -s + n_b)

    return (a_start, a_end), (b_start, b_end)


@export
def split_touching_windows(things, containers, window=0):
    """Split things by their containers and return a list of length containers.

    :param things: Sorted array of interval-like data
    :param containers: Sorted array of interval-like data
    :param window: threshold distance for touching check.

    For example:
        - window = 0: things must overlap one sample
        - window = -1: things can start right after container ends
            (i.e. container endtime equals the thing starttime, since strax
            endtimes are exclusive)
    :return:

    """
    windows = touching_windows(things, containers, window)
    return _split_by_window(things, windows)


@numba.njit
def _split_by_window(r, windows):
    result = []
    for w in windows:
        result.append(r[w[0] : w[1]])
    return result


@export
def touching_windows(things, containers, window=0):
    """Return array of (start, exclusive end) indices into things which extend to within window of
    the container, for each container in containers.

    :param things: Sorted array of interval-like data.
        We assume all things and containers are sorted by time.
        When endtime are not sorted, it will return indices
        of the first and last things which are touching the container.
    :param containers: Sorted array of interval-like data. Containers are
        allowed to overlap.
    :param window: threshold distance for touching check.

    For example:
        - window = 0: things must overlap one sample
        - window = -1: things can start right after container ends
            (i.e. container endtime equals the thing starttime, since strax
            endtimes are exclusive)

    """
    try:
        _check_time_is_sorted(things["time"])
    except Exception:
        raise ValueError("time of things should be sorted!")
    try:
        _check_time_is_sorted(strax.endtime(things))
    except Exception:
        warnings.warn(
            "endtime of things is not sorted! "
            "touching_windows will return the indices of the "
            "first and last things which are touching the container."
        )
    try:
        _check_time_is_sorted(containers["time"])
    except Exception:
        raise ValueError("time of containers should be sorted!")
    for objects, names in zip([things, containers], ["things", "containers"]):
        try:
            _check_objects_non_negative_length(objects)
        except Exception:
            raise ValueError(f"{names} should have non-negative length!")

    # return zeros if either things or containers are empty
    if len(things) == 0 or len(containers) == 0:
        return np.zeros((len(containers), 2), dtype=np.int32)

    return _touching_windows(
        things["time"],
        strax.endtime(things),
        containers["time"],
        strax.endtime(containers),
        window=window,
    )


@numba.njit(nogil=True, cache=True)
def _touching_windows(
    thing_start, thing_end, container_start, container_end, window=0, endtime_sort_kind="mergesort"
):
    n = len(thing_start)
    container_end_argsort = stable_argsort(container_end, kind=endtime_sort_kind)

    # we search twice, first for the beginning of the interval, then for the end
    left_i = right_i = 0
    result = np.zeros((len(container_start), 2), dtype=np.int32)

    # first search for the beginning of the interval
    # containers' time is already sorted, but things' endtime is not
    for i, t0 in enumerate(container_start):
        while left_i <= n - 1 and thing_end[left_i] <= t0 - window:
            # left_i ends before the window starts (so it's still outside)
            left_i += 1
        # save the most left index of things touching the container
        result[i, 0] = left_i

    # then search for the end of the interval
    # containers' endtime is not sorted but things' endtime is
    for i in container_end_argsort:
        t1 = container_end[i]
        while right_i <= n - 1 and thing_start[right_i] < t1 + window:
            # right_i starts before the window ends (so it could be inside)
            right_i += 1
        # now right_i is the last index inside the window or outside the array
        result[i, 1] = right_i

    return result


@numba.njit(nogil=True, cache=True)
def _check_time_is_sorted(time):
    """Check if times are sorted."""
    mask = np.all((time[1:] - time[:-1]) >= 0)
    assert mask


@numba.njit(nogil=True, cache=True)
def _check_objects_non_negative_length(objects):
    """Checks if objects have non-negative length."""
    mask = np.all(strax.endtime(objects) >= objects["time"])
    assert mask


@numba.njit(nogil=True, cache=True)
def _check_objects_are_not_overlapping(objects):
    """Checks if objects overlap in time."""
    mask = np.all(objects["time"][1:] - strax.endtime(objects)[:-1] >= 0)
    assert mask


@export
def abs_time_to_prev_next_interval(things, intervals):
    """Function which determines the time difference of things to previous and next interval, e.g.,
    events to veto intervals. Assumes that things do not overlap.

    :param things: Numpy structured array containing strax time fields
    :param intervals: Numpy structured array containing time fields
    :return: Two integer arrays with the time difference to the previous and next intervals.

    """
    try:
        _check_time_is_sorted(things["time"])
    except Exception:
        raise ValueError("time of things should be sorted!")
    try:
        _check_time_is_sorted(strax.endtime(things))
    except Exception:
        warnings.warn(
            "endtime of things is not sorted! "
            "times_to_next returned by abs_time_to_prev_next_interval "
            "might be larger than expected."
        )
    try:
        _check_time_is_sorted(intervals["time"])
    except Exception:
        raise ValueError("time of intervals should be sorted!")

    times_to_prev = np.ones(len(things), dtype=np.int64) * -1
    times_to_next = np.ones(len(things), dtype=np.int64) * -1

    _empty_events_or_intervals = (len(things) == 0) or (len(intervals) == 0)
    if _empty_events_or_intervals:
        return times_to_prev, times_to_next

    _abs_time_to_prev_next(things, intervals, times_to_prev, times_to_next)

    return times_to_prev, times_to_next


@numba.njit
def _abs_time_to_prev_next(things, intervals, times_to_prev, times_to_next):
    veto_intervals_seen = 0
    for thing_ind, thing_i in enumerate(things):
        current_event_time = thing_i["time"]
        current_event_endtime = strax.endtime(thing_i)

        # Exploit the fact that events cannot overlap...
        # Loop over veto intervals until:
        #   - current veto interval endtime starts overlapping with
        #    current event. This is the interval from which we need to
        #    start looping for the next event.
        #   - current veto_interval time starts to be larger than current
        #    event time. Only then we can be sure we have computed the
        #    shortest time delay as endtime is not sorted...
        for veto_interval in intervals[veto_intervals_seen:]:
            _interval_start_after_thing = veto_interval["time"] >= current_event_time
            if _interval_start_after_thing:
                break

            # Always update endtime until it becomes negative:
            dt = current_event_time - strax.endtime(veto_interval)
            _ends_befor_event = dt >= 0
            if _ends_befor_event:
                times_to_prev[thing_ind] = dt
                veto_intervals_seen += 1

        # Now check if current veto is still within event or already after it:
        for veto_interval in intervals[veto_intervals_seen:]:
            _current_interval_before_thing_ends = veto_interval["time"] < current_event_endtime
            if _current_interval_before_thing_ends:
                continue

            # Now current veto is after event so store time:
            times_to_next[thing_ind] = veto_interval["time"] - current_event_endtime
            break

        veto_intervals_seen = max(0, veto_intervals_seen - 1)
