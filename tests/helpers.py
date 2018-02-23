from itertools import accumulate
from functools import partial

import numpy as np
from boltons import iterutils
from hypothesis import strategies as st

import strax


# Since we use np.cumsum to get disjoint intervals, we don't want stuff
# wrapping around to the integer boundary


def sorted_bounds(disjoint=False,
                  max_value=10000,
                  max_len=100,
                  remove_duplicates=False):
    if disjoint:
        # Since we accumulate later:
        max_value /= max_len

    s = st.lists(st.integers(min_value=0,
                             max_value=max_value),
                 min_size=0, max_size=20)
    if disjoint:
        s = s.map(accumulate).map(list)

    # Select only cases with even-length lists
    s = s.filter(lambda x: len(x) % 2 == 0)

    # Convert to list of 2-tuples
    s = s.map(lambda x: [tuple(q)
                         for q in iterutils.chunked(sorted(x), size=2)])

    # Remove cases with zero-length intervals
    s = s.filter(lambda x: all([a[0] != a[1] for a in x]))

    if remove_duplicates:
        # (this will always succeed if disjoint=True)
        s = s.filter(lambda x: x == list(set(x)))

    # Sort intervals and result
    return s.map(sorted)


##
# Fake intervals
##

def bounds_to_intervals(bs, dtype=strax.interval_dtype):
    x = np.zeros(len(bs), dtype=dtype)
    x['time'] = [x[0] for x in bs]
    # Remember: exclusive right bound...
    x['length'] = [x[1] - x[0] for x in bs]
    x['dt'] = 1
    return x


sorted_intervals = sorted_bounds().map(bounds_to_intervals)

disjoint_sorted_intervals = sorted_bounds(disjoint=True).\
    map(bounds_to_intervals)

fake_hits = sorted_bounds().map(partial(bounds_to_intervals,
                                        dtype=strax.hit_dtype))

##
# Fake pulses with 0 or 1 as waveform (e.g. to test hitfinder)
##

def bounds_to_pulse(bs):
    if not len(bs):
        n_samples = 0
    else:
        n_samples = max([a for b in bs for a in b])
        if n_samples % 2:
            # Make sure we sometimes end in zero
            # TODO: not a great way to do it, you miss other cases..
            n_samples += 1
    rec = np.zeros(1, dtype=strax.record_dtype(n_samples))
    rec['dt'] = 1
    for l, r in bs:
        rec[0]['data'][l:r] = 1
    return rec


single_fake_pulse = sorted_bounds(max_value=50).map(bounds_to_pulse)
