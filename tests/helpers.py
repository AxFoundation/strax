import tempfile     # noqa
from itertools import accumulate
from functools import partial

import numpy as np
from boltons import iterutils
from hypothesis import strategies
import pytest     # noqa

##
# Hack to disable numba.jit
# For the mini-examples run during testing numba actually causes a massive
# performance drop. Moreover, if you make a buffer overrun bug, jitted fs
# respond "slightly" less nice (giving you junk data or segfaulting)
# Once in a while you should test without this...
##
def mock_numba():
    from unittest.mock import MagicMock

    class FakeNumba:

        def jit(self, *args, **kwargs):
            return lambda x: x

    FakeNumba.caching = MagicMock()

    import sys                                # noqa
    sys.modules['numba'] = FakeNumba()


# Mock numba before importing strax
mock_numba()
import strax    # noqa


# Since we use np.cumsum to get disjoint intervals, we don't want stuff
# wrapping around to the integer boundary. Hence max_value is limited.
def sorted_bounds(disjoint=False,
                  max_value=50,
                  max_len=10,
                  remove_duplicates=False):
    if disjoint:
        # Since we accumulate later:
        max_value /= max_len

    s = strategies.lists(strategies.integers(min_value=0,
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

# TODO: isn't this duplicated with bounds_to_records??

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


def bounds_to_records(bs, single=False, single_channel=False):
    """Return strax records corresponding to a list of 2-tuples
    of boundaries.

    By default, for each boundary tuple, create a pulse whose data is 1 inside.
    The pulses are put in different channels, first in 0, second in 1, etc.

    :param single: if True, instead create a single pulse in channel 0
    whose data is 1 inside the given bounds and zero outside.
    TODO: length etc. is not properly set in the single=True mode!
    TODO: this probably needs tests itself...

    :param single_channel: if True, instead create all pulses in channel 0
    You should only feed in disjoint bounds when using this.
    """
    if not len(bs):
        n_samples = 0
    else:
        n_samples = max([a for b in bs for a in b])
        if n_samples % 2:
            # Make sure we sometimes end in zero
            # TODO: not a great way to do it, you miss other cases..
            n_samples += 1
    if not single:
        # Each bound gets its own pulse, in its own channel
        recs = np.zeros(len(bs), dtype=strax.record_dtype(n_samples))
        for i, (l, r) in enumerate(bs):
            # Add waveform roughly in the center
            length = r - l  # Exclusive right bound, no + 1
            pad = (n_samples - (r - l)) // 2
            recs[i]['time'] = l
            recs[i]['length'] = pad + length
            recs[i]['data'][pad:pad+length] = 1
            assert recs[i]['data'].sum() == length
            recs[i]['channel'] = 0 if single_channel else i
        if not single_channel:
            assert len(np.unique(recs['channel'])) == len(bs)
    else:
        # Make a single record with 1 inside the bounds, 0 outside
        recs = np.zeros(1, dtype=strax.record_dtype(n_samples))
        for l, r in bs:
            recs[0]['data'][l:r] = 1
        recs[0]['time'] = 0
        recs[0]['length'] = n_samples

    recs['dt'] = 1
    return recs


single_fake_pulse = sorted_bounds()\
    .map(partial(bounds_to_records, single=True))

several_fake_records = sorted_bounds().map(bounds_to_records)

several_fake_records_one_channel = sorted_bounds(
    disjoint=True).map(
        partial(bounds_to_records, single_channel=True))


##
# Basic test plugins
##
@strax.takes_config(
    strax.Option('crash', default=False),
)
class Records(strax.Plugin):
    provides = 'records'
    parallel = 'process'
    depends_on = tuple()
    dtype = strax.record_dtype()

    def source_finished(self):
        return True

    def is_ready(self, chunk_i):
        return chunk_i < n_chunks

    def compute(self, chunk_i):
        if self.config['crash']:
            raise SomeCrash("CRASH!!!!")
        r = np.zeros(recs_per_chunk, self.dtype)
        r['time'] = chunk_i
        r['length'] = 1
        r['dt'] = 1
        r['channel'] = np.arange(len(r))
        return r


class SomeCrash(Exception):
    pass


@strax.takes_config(
    strax.Option('some_option', default=0),
    strax.Option('give_wrong_dtype', default=False)
)
class Peaks(strax.Plugin):
    provides = 'peaks'
    depends_on = ('records',)
    dtype = strax.peak_dtype()
    parallel = True

    def compute(self, records):
        if self.config['give_wrong_dtype']:
            return np.zeros(5, [('a', np.int), ('b', np.float)])
        p = np.zeros(len(records), self.dtype)
        p['time'] = records['time']
        return p


recs_per_chunk = 10
n_chunks = 10
run_id = '0'
