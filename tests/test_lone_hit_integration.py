from strax.testutils import several_fake_records
import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as st

import strax


@settings(deadline=None)
@given(several_fake_records,
       st.integers(min_value=0, max_value=100),
       st.integers(min_value=0, max_value=100),
       )
def test_lone_hits_integration_bounds(records, left_extension, right_extension):
    """
    Loops over hits and tests if integration bounds overlap.
    """
    n_channel = 0
    if len(records):
        n_channel = records['channel'].max()+1

    hits = strax.find_hits(records, np.ones(n_channel))

    strax.find_hit_integration_bounds(hits,
                                      np.zeros(0, dtype=strax.time_dt_fields),
                                      records,
                                      (left_extension, right_extension),
                                      n_channel,
                                      allow_bounds_beyond_records=False
                                      )
    _test_overlap(hits)

    hits['left_integration'] = 0
    hits['right_integration'] = 0

    strax.find_hit_integration_bounds(hits,
                                      np.zeros(0, dtype=strax.time_dt_fields),
                                      records,
                                      (left_extension, right_extension),
                                      n_channel,
                                      allow_bounds_beyond_records=True
                                      )
    _test_overlap(hits)


def _test_overlap(hits):
    tester = np.zeros(len(hits), dtype=strax.time_fields)
    tester['time'] = hits['time'] - (hits['left_integration'] - hits['left'])*hits['dt']
    tester['endtime'] = hits['time'] + (hits['right_integration'] - hits['left'])*hits['dt']

    for ch in np.unique(hits['channel']):
        mask = hits['channel'] == ch
        test_ch = np.all((tester[mask]['endtime'][:-1] - tester[mask]['time'][1:]) <= 0)
        assert np.all(test_ch), 'Hits overlap!'
