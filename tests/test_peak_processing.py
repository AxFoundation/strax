from strax.testutils import fake_hits, several_fake_records

import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as st

import strax


@given(fake_hits,
       st.one_of(st.just(1), st.just(3)),
       st.one_of(st.just(0), st.just(3)))
@settings(deadline=None)
def test_find_peaks(hits, min_channels, min_area):
    hits['area'] = 1
    gap_threshold = 10
    peaks = strax.find_peaks(hits,
                             adc_to_pe=np.ones(1),
                             right_extension=0, left_extension=0,
                             gap_threshold=gap_threshold,
                             min_channels=min_channels,
                             min_area=min_area)
    # Check sanity
    assert np.all(peaks['length'] > 0)
    assert np.all(peaks['n_hits'] > 0)

    # Check if requirements satisfied
    if min_area != 0:
        assert np.all(peaks['area'] >= min_area)
    if min_channels != 1:
        assert np.all(peaks['n_hits'] >= min_channels)
    assert np.all(peaks['max_gap'] < gap_threshold)

    # Without requirements, all hits must occur in a peak
    if min_area == 0 and min_channels == 1:
        assert np.sum(peaks['n_hits']) == len(hits)
        assert np.all(strax.fully_contained_in(hits, peaks) > -1)

    # Since no extensions, peaks must be at least gap_threshold apart
    starts = peaks['time']
    ends = peaks['time'] + peaks['length'] * peaks['dt']
    assert np.all(ends[:-1] + gap_threshold <= starts[1:])

    assert np.all(starts == np.sort(starts)), "Not sorted"

    assert np.all(peaks['time'] < strax.endtime(peaks)), "Non+ peak length"

    # TODO: add more tests, preferably test against a second algorithm


@settings(deadline=None)
@given(several_fake_records,
       st.integers(min_value=0, max_value=100),
       st.integers(min_value=1, max_value=100)
       )
def test_sum_waveform(records, peak_left, peak_length):
    # Make a single big peak to contain all the records
    n_ch = 100
    peaks = np.zeros(1, strax.peak_dtype(n_ch, n_sum_wv_samples=200))
    p = peaks[0]
    p['time'] = peak_left
    p['length'] = peak_length
    p['dt'] = 1

    strax.sum_waveform(peaks, records, np.ones(n_ch))

    # Area measures must be consistent
    area = p['area']
    assert area >= 0
    assert p['data'].sum() == area
    assert p['area_per_channel'].sum() == area

    # Create a simple sum waveform
    if not len(records):
        max_sample = 3   # Whatever
    else:
        max_sample = (records['time'] + records['length']).max()
    max_sample = max(max_sample, peak_left + peak_length)
    sum_wv = np.zeros(max_sample + 1, dtype=np.float32)
    for r in records:
        sum_wv[r['time']:r['time'] + r['length']] += r['data'][:r['length']]
    # Select the part inside the peak
    sum_wv = sum_wv[peak_left:peak_left + peak_length]

    assert len(sum_wv) == peak_length
    assert np.all(p['data'][:peak_length] == sum_wv)

    # Finally check that we also can use a selection of peaks to sum
    strax.sum_waveform(peaks, records, np.ones(n_ch), select_peaks_indices=np.array([0]))
