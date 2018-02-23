import numpy as np
from hypothesis import given

import strax
from .helpers import fake_hits


@given(fake_hits)
def test_find_peaks(hits):
    gap_threshold = 10
    peaks = strax.find_peaks(hits,
                             to_pe=np.ones(1),
                             right_extension=0, left_extension=0,
                             gap_threshold=gap_threshold,
                             min_hits=1)

    # Since min_hits = 1, all hits must occur in a peak
    assert np.sum(peaks['n_hits']) == len(hits)
    assert np.all(strax.fully_contained_in(hits, peaks) > -1)

    # Since no extensions, peaks must be at least gap_threshold apart
    starts = peaks['time']
    ends = peaks['time'] + peaks['length'] * peaks['dt']
    assert np.all(ends[:-1] + gap_threshold <= starts[1:])

    assert np.all(starts == np.sort(starts))

    # TODO: add more tests, preferably test against a second algorithm
