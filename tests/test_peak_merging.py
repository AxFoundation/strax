import hypothesis
import numpy as np

import strax
from strax.testutils import disjoint_sorted_intervals, fake_hits, sorted_bounds, bounds_to_intervals
from functools import partial


@hypothesis.given(disjoint_sorted_intervals,
                  disjoint_sorted_intervals)
@hypothesis.settings(max_examples=1000, deadline=None)
def test_replace_merged(intervals, merge_instructions):
    # First we have to create some merged intervals.
    # We abuse the interval generation mechanism to create 'merge_instructions'
    # i.e. something to tell us which indices of intervals must be merged
    # together.

    merged_itvs = []
    to_remove = []
    for x in merge_instructions:
        start, end_inclusive = x['time'], x['time'] + x['length'] - 1
        if end_inclusive == start or end_inclusive >= len(intervals):
            # Pointless / invalid merge instruction
            continue
        to_remove.extend(list(range(start, end_inclusive + 1)))
        new = np.zeros(1, strax.interval_dtype)[0]
        new['time'] = intervals[start]['time']
        new['length'] = strax.endtime(intervals[end_inclusive]) - new['time']
        new['dt'] = 1
        merged_itvs.append(new)
    removed_itvs = []
    kept_itvs = []
    for i, itv in enumerate(intervals):
        if i in to_remove:
            removed_itvs.append(itv)
        else:
            kept_itvs.append(itv)

    kept_itvs = np.array(kept_itvs)
    merged_itvs = np.array(merged_itvs)

    result = strax.replace_merged(intervals, merged_itvs)
    assert len(result) == len(merged_itvs) + len(kept_itvs)
    assert np.all(np.diff(result['time']) > 0), "Not sorted"
    assert np.all(result['time'][1:] - strax.endtime(result)[:-1] >= 0), "Overlap"
    for x in kept_itvs:
        assert x in result, "Removed too many"
    for x in merged_itvs:
        assert x in result, "Didn't put in merged"
    for x in result:
        assert np.isin(x, merged_itvs) or np.isin(x, kept_itvs), "Invented itv"


def bounds_to_intervals_w_data(bs, dt_max=10, dtype=strax.interval_dtype):
    """Similar to bounds_to_intervals from strax.testutils but with a data field"""
    x = np.zeros(len(bs), dtype=dtype)
    x['time'] = [x[0] for x in bs]
    x['dt'] = np.random.randint(1, dt_max, size=len(x))
    # Remember: exclusive right bound...
    x['length'] = [(x[1] - x[0]) // dt for x, dt in zip(bs, x['dt'])]
    # Clip length to be at least shorter than the 'data' field
    x['length'] = np.clip(x['length'], 0, x['data'].shape[1])
    for x_i in x:
        x_i['data'][:x_i['length']] = np.random.random(size=x_i['length'])
    return x


def get_peaks(min_peaks=2, **dtype_kwargs):
    """return some fake peaks"""
    s = sorted_bounds(min_size=min_peaks, disjoint=True).map(
        partial(bounds_to_intervals_w_data, dtype=strax.peak_dtype(**dtype_kwargs)))
    # Make sure we got at least <min_peaks> after the filtering
    s = s.filter(lambda x: len(x) >= min_peaks)
    return s


@hypothesis.example(
    peaks=np.array([
        (0, 0, 6, 0, 0, [0., 0., 0.], 0., [], 0, [0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [], 0, 0, 0, 0.),
        (2, 3, 1, 0, 0, [0., 0., 0.], 0., [], 0, [0.60276335, 0.5448832, 0.4236548],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [], 0, 0, 0, 0.)],
        dtype=[(('Start time since unix epoch [ns]', 'time'), '<i8'),
               (('Length of the interval in samples', 'length'), '<i4'),
               (('Width of one sample [ns]', 'dt'), '<i4'),
               (('Channel/PMT number', 'channel'), '<i2'),
               (('Classification of the peak(let)', 'type'), 'i1'),
               (('Waveform data in PE/sample (not PE/ns!), top array', 'data_top'), '<f4', (3,)),
               (('Integral across channels [PE]', 'area'), '<f4'),
               (('Integral per channel [PE]', 'area_per_channel'), '<f4', (0,)),
               (('Number of hits contributing at least one sample to the peak ', 'n_hits'), '<i4'),
               (('Waveform data in PE/sample (not PE/ns!)', 'data'), '<f4', (3,)),
               (('Peak widths in range of central area fraction [ns]', 'width'), '<f4', (11,)),
               (('Peak widths: time between nth and 5th area decile [ns]', 'area_decile_from_midpoint'), '<f4', (11,)),
               (('Does the channel reach ADC saturation?', 'saturated_channel'), 'i1', (0,)),
               (('Total number of saturated channels', 'n_saturated_channels'), '<i2'),
               (('Channel within tight range of mean', 'tight_coincidence'), '<i2'),
               (('Largest gap between hits inside peak [ns]', 'max_gap'), '<i4'),
               (('Maximum interior goodness of split', 'max_goodness_of_split'), '<f4')])
)
@hypothesis.settings(deadline=None, max_examples=1_000)
@hypothesis.given(get_peaks(n_sum_wv_samples=3, n_channels=0))
def test_data_field(peaks):
    """
    Test https://github.com/AxFoundation/strax/issues/704
    merge all the peaks - which should give us a similarly large peak as all the input peaks
    """
    new_peaks = strax.merge_peaks(peaks,
                                  np.array([0]),
                                  np.array([len(peaks)]),
                                  len(peaks['data']) * len(peaks) * 10
                                  )

    np.testing.assert_allclose(peaks['data'].sum(), new_peaks['data'].sum(), atol=0, rtol=1e-6)


@hypothesis.given(fake_hits,
                  hypothesis.strategies.integers(min_value=0, max_value=int(1e18)),
                  hypothesis.strategies.integers(min_value=0, max_value=100),
                  hypothesis.strategies.integers(min_value=1, max_value=2),
                  )
@hypothesis.settings(deadline=None)
def test_add_lone_hits(hits, time_offset, peak_length, dt):
    peak = np.zeros(1, dtype=strax.peak_dtype())
    peak['time'] = time_offset
    hits['time'] += time_offset
    peak['length'] = peak_length
    hits['area'] = 1
    peak['dt'] = dt

    to_pe = np.ones(10000)
    strax.add_lone_hits(peak, hits, to_pe)

    if not peak_length:
        assert peak['area'] == 0
        assert peak['data'].sum() == 0
        return

    split_hits = strax.split_by_containment(hits, peak)[0]
    dummy_peak = np.zeros(peak_length)

    for h in split_hits:
        dummy_peak[(h['time']-time_offset)//dt] += h['area']
    peak = peak[0]
    assert peak['area'] == np.sum(split_hits['area'])
    assert np.all(peak['data'][:peak_length] == dummy_peak)
