import strax
import numba
import numpy as np

export, __all__ = strax.exporter()


# (5-10x) faster than np.sort(order=...), as np.sort looks at all fields
# TODO: maybe this should be a factory?
@export
@numba.jit(nopython=True, nogil=True, cache=True)
def sort_by_time(x):
    """Sort pulses by time, then channel.

    Assumes you have no more than 10k channels, and records don't span
    more than 100 days. TODO: FIX this
    """
    if len(x) == 0:
        # Nothing to do, and .min() on empty array doesn't work, so:
        return x
    # I couldn't get fast argsort on multiple keys to work in numba
    # So, let's make a single key...
    sort_key = (x['time'] - x['time'].min()) * 10000 + x['channel']
    sort_i = np.argsort(sort_key)
    return x[sort_i]


@export
@numba.jit(nopython=True, nogil=True, cache=True)
def first_index_not_below(arr, t):
    """Return first index of array >= t, or len(arr) if no such found"""
    for i, x in enumerate(arr):
        if x >= t:
            return i
    return len(arr)


@export
def endtime(x):
    """Return endtime of intervals x"""
    try:
        return x['endtime']
    except (KeyError, ValueError, IndexError):
        return x['time'] + x['length'] * x['dt']


@export
# TODO: somehow numba compilation hangs on this one? reproduce / file issue?
# numba.jit(nopython=True, nogil=True, cache=True)
def from_break(x, safe_break=10000, left=True, tolerant=False):
    """Return records on side of a break at least safe_break long
    If there is no such break, return the best break found.
    """
    # TODO: This is extremely rough. Better to find proper gaps, and if we
    # know the timing of the readers, consider breaks at end and start too.
    break_i = find_break_i(x, safe_break=safe_break, tolerant=tolerant)

    if left:
        return x[:break_i]
    else:
        return x[break_i:]


@export
class NoBreakFound(Exception):
    pass


@export
@numba.jit(nopython=True, nogil=True, cache=True)
def find_break_i(x, safe_break, tolerant=True):
    """Returns LAST index of x whose time is more than safe_break away
    from the x before
    :param tolerant: if no break found, yield an as good as possible break
    anyway.
    """
    max_gap = 0
    max_gap_i = -1
    for _i in range(len(x) - 1):
        i = len(x) - 1 - _i
        gap = x[i]['time'] - x[i - 1]['time']
        if gap >= safe_break:
            return i
        if gap > max_gap:
            max_gap_i = i
            max_gap = gap

    if not tolerant:
        raise NoBreakFound

    print("\t\tDid not find safe break, using largest available break: ",
          max_gap,
          " ns")
    return max_gap_i


@export
def fully_contained_in(things, containers):
    """Return array of len(things) with index of interval in containers
    for which things are fully contained in a container, or -1 if no such
    exists.
    We assume all intervals are sorted by time, and b_intervals
    nonoverlapping.
    """
    result = np.ones(len(things), dtype=np.int32) * -1
    a_starts = things['time']
    b_starts = containers['time']
    a_ends = strax.endtime(things)
    b_ends = strax.endtime(containers)
    _fc_in(a_starts, b_starts, a_ends, b_ends, result)
    return result


@numba.jit(nopython=True, nogil=True, cache=True)
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
    """Return list of thing-arrays contained in each container

    Assumes everything is sorted, and containers are nonoverlapping
    """
    if not len(containers):
        return []

    # Index of which container each thing belongs to, or -1
    which_container = fully_contained_in(things, containers)

    # Restrict to things in containers
    mask = which_container != -1
    things = things[mask]
    which_container = which_container[mask]
    if not len(things):
        # np.split has confusing behaviour for empty arrays
        return [things[:0] for _ in range(len(containers))]

    # Split things up by container
    split_indices = np.where(np.diff(which_container))[0] + 1
    things_split = np.split(things, split_indices)

    # Insert empty arrays for empty containers
    empty_containers = np.setdiff1d(np.arange(len(containers)),
                                    np.unique(which_container))
    for c_i in empty_containers:
        things_split.insert(c_i, things[:0])

    return things_split
