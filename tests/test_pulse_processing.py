from .helpers import single_fake_pulse

import numpy as np
from hypothesis import given
from scipy.ndimage import convolve1d

import strax


def _find_hits(r):
    hits = strax.find_hits(r, threshold=0)
    # Test pulses have dt=1 and time=0
    # TODO: hm, maybe this doesn't test everything
    np.testing.assert_equal(hits['time'], hits['left'])
    # NB: exclusive right bound, no + 1 here
    np.testing.assert_equal(hits['length'],
                            hits['right'] - hits['left'])
    for h in hits:
        q = r[h['record_i']]
        assert q['data'][h['left']:h['right']].sum() == h['area']
    return list(zip(hits['left'], hits['right']))


def test_find_hits():
    """Tests the hitfinder with simple example pulses"""
    for w, should_find_intervals in [
            ([], []),
            ([1], [(0, 1)]),
            ([1, 0], [(0, 1)]),
            ([1, 0, 1], [(0, 1), (2, 3)]),
            ([1, 0, 1, 0], [(0, 1), (2, 3)]),
            ([1, 0, 1, 0, 1], [(0, 1), (2, 3), (4, 5)]),
            ([0, 1, 2, 0, 4, -1, 60, 700, -4], [(1, 3), (4, 5), (6, 8)]),
            ([1, 1, 2, 0, 4, -1, 60, 700, -4], [(0, 3), (4, 5), (6, 8)]),
            ([1, 0, 2, 3, 4, -1, 60, 700, -4], [(0, 1), (2, 5), (6, 8)]),
            ([1, 0, 2, 3, 4, -1, 60, 700, 800], [(0, 1), (2, 5), (6, 9)]),
            ([0, 0, 2, 3, 4, -1, 60, 700, 800], [(2, 5), (6, 9)])]:

        records = np.zeros(1, strax.record_dtype(9))
        records[0]['data'][:len(w)] = w
        records['dt'] = 1
        records['length'] = 9

        results = _find_hits(records)
        assert len(results) == len(should_find_intervals)
        assert results == should_find_intervals


@given(single_fake_pulse)
def test_find_hits_randomize(records):
    """Tests the hitfinder with whatever hypothesis can throw at it
    (well, pulse only takes (0, 1), and we only test a single pulse at a time)
    """
    results = _find_hits(records)
    w = records[0]['data']

    # Check for false positives
    for l, r in results:
        assert np.all(w[l:r] == 1)

    # Check for false negatives
    for i in range(len(results) - 1):
        l_ = results[i][1]
        r_ = results[i + 1][0]
        assert not np.any(w[l_:r_] == 1)


def test_filter_waveforms():
    """Test that filter_records gives the same output
    as a simple convolution applied to the original pulse
    (before splitting into records)
    """
    wv = np.random.randn(300)
    ir = np.random.randn(41)
    ir[10] += 10   # Because it crashes for max at edges
    origin = np.argmax(ir) - (len(ir)//2)
    wv_after = convolve1d(wv, ir,
                          mode='constant',
                          origin=origin)

    wvs = wv.reshape(3, 100)
    wvs = strax.filter_waveforms(
        wvs, ir,
        prev_r=np.array([strax.NO_RECORD_LINK, 0, 1]),
        next_r=np.array([1, 2, strax.NO_RECORD_LINK]))
    wv_after_2 = np.reshape(wvs, -1)

    assert np.abs(wv_after - wv_after_2).sum() < 1e-9
