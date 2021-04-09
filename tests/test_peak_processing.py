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
        max_sample = 3  # Whatever
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


@settings(deadline=None)
@given(several_fake_records,
       st.integers(min_value=1, max_value=10000),
       st.integers(min_value=10, max_value=4000),
       st.integers(min_value=int(1e3), max_value=int(7_000_000)),
       st.integers(min_value=0, max_value=int(1e4)),
       st.integers(min_value=int(0), max_value=int(1e4)),
       st.integers(min_value=int(1e4), max_value=int(2e4)),
       )
def test_peak_overflow(records,
                       gap_factor,
                       record_repetition_factor,
                       max_duration,
                       left_extension,
                       right_extension,
                       gap_threshold,
                       ):
    """
    Test that we handle dt overflows in peaks correctly. To this end, we
        just create some sets of records and copy that set of records
        for a few times. That way we may end up with a very long
        artificial set of hits that can be used in the peak building. By
        setting the peak finding parameters to very strange conditions
        we are able to replicate the behaviour where a peak would become
        so large that it cannot be written out correctly due to integer
        overflow of the dt field,
    :param records: records
    :param gap_factor: to create very extended sets of records, just
        add a factor that can be used to multiply the time field with,
        to more quickly arrive to a very long pulse-train
    :param record_repetition_factor: how much do we repeat the records
        pusle train
    :param max_duration: max_duration option for strax.find_peaks
    :param left_extension: option for strax.find_peaks
    :param right_extension: option for strax.find_peaks
    :param gap_threshold: option for strax.find_peaks
    :return: None
    """
    p = np.zeros(0, dtype=strax.peak_dtype())
    magic_overflow_time = np.iinfo(p.dtype['dt']).max * p.dtype['data'].shape[0]
    del p

    # Make a single big peak to contain all the records
    def retrun_1(x):
        """
        Return 1 for all of the input that can be used as a parameter
            for the splitting in natural breaks
        :param x: any type of array
        :return: ones * len(array)
        """
        ret = np.ones(len(x))
        return ret

    r = records
    if not len(r):
        # Hard to test integer overflow for empty records
        return

    # Copy the pulse train of the records. We are going to copy the same
    # set of records many times now
    r_buffer = []
    t_max = strax.endtime(r)
    for i in range(record_repetition_factor):
        r_copy = r.copy()
        r_copy['time'] = r_copy['time'] + t_max * i * gap_factor
        r_buffer.append(r_copy)
        if r_copy['time'][-1] - r['time'][0] > 2 * magic_overflow_time:
            # No need to go over and beyond
            break
    r = np.concatenate(r_buffer)

    # Do peak finding!
    hits = strax.find_hits(r, min_amplitude=0)
    assert len(hits)
    hits = strax.sort_by_time(hits)

    # Dummy to_pe
    to_pe = np.ones(max(r['channel']))

    # Find peaks, we might end up with negative dt here!
    p = strax.find_peaks(hits, to_pe,
                         gap_threshold=gap_threshold,
                         left_extension=left_extension,
                         right_extension=right_extension,
                         # max_duration=max_duration,
                         # Due to these settings, we will start merging
                         # whatever strax can get its hands on
                         min_area=0,
                         min_channels=1, )
    assert len(p)
    assert np.all(p['dt'] > 0)

    # Compute basics
    strax.sum_waveform(p, r, to_pe)
    strax.compute_widths(p)
    try:
        peaklets = strax.split_peaks(
            p, r, to_pe,
            algorithm='natural_breaks',
            threshold=retrun_1,
            split_low=True,
            filter_wing_width=70,
            min_area=0,
            do_iterations=2)
    except AssertionError as e:
        if left_extension+max_duration+right_extension > magic_overflow_time:
            print(f'Great, the test worked, we are getting the assertion statement')
        else:
            raise e
    finally:
        if left_extension + max_duration + right_extension > magic_overflow_time:
            raise ValueError(
                'We were not properly warned of the imminent peril we are '
                'facing. Where is our white knight in shining armour to '
                'protected from imminent doom')
    assert len(peaklets)
    assert len(peaklets) <= len(records)
    assert peaklets['dt']
