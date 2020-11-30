import strax
import numpy as np
from hypothesis import given, strategies, example, settings
import tempfile


@settings(max_examples=100, deadline=None)
@given(
    # number of peaks
    strategies.integers(min_value=0, max_value=20),
    # length of the data field in the peaks
    strategies.integers(min_value=2, max_value=20),
)
def test_index_of_fraction(peak_length, data_length):
    """
    Test strax.index_of_fraction
    """
    # Setup the peaks
    dtype = [(('Start time since unix epoch [ns]', 'time'), np.int64),
             (('Exclusive end time since unix epoch [ns]', 'endtime'), np.int64),
             (('length of p', 'length'), np.int16),
             (('area of p', 'area'), np.float64),
             (('data of p', 'data'), (np.float64, data_length))]
    peaks = np.zeros(peak_length, dtype=dtype)
    peaks['time'] = np.arange(peak_length)
    peaks['endtime'] = peaks['time'] + 1

    # Fill the peaks with random length data
    for p in peaks:
        length = np.random.randint(0, data_length)
        p['length'] = length
        wf = np.random.random(size=data_length)
        p['data'] = wf

        for i in range(length, len(wf)):
            # Set all the out of length data to zero
            wf[i] = 0.

        if len(peaks):
            # Compute sum area
            peaks['area'] = np.sum(peaks['data'], axis=1)

    fraction_desired = np.random.random(size=peak_length)
    res = strax.index_of_fraction(peaks, fraction_desired)
    assert len(res) == len(peaks), "Lost peaks"
    if len(peaks):
        assert np.max(res) <= data_length, "Index returned out of bound"


@settings(max_examples=100, deadline=None)
@given(
    # number of peaks
    strategies.integers(min_value=0, max_value=20),
    # length of the data field in the peaks
    strategies.integers(min_value=2, max_value=20),
    # Number of widths to compute
    strategies.integers(min_value=2, max_value=10),
)
def test_compute_widths(peak_length, data_length, n_widths):
    """
    Test strax.compute_widths
    """
    dtype = [(('Start time since unix epoch [ns]', 'time'), np.int64),
             (('dt in ns', 'dt'), np.int64),
             (('length of p', 'length'), np.int16),
             (('area of p', 'area'), np.float64),
             (('data of p', 'data'), (np.float64, data_length)),
             (('width of p', 'width'), (np.float64, n_widths)),
             (('area_decile_from_midpoint of p', 'area_decile_from_midpoint'), (np.float64, n_widths))]
    peaks = np.zeros(peak_length, dtype=dtype)
    dt = 1
    peaks['time'] = np.arange(peak_length)*dt
    peaks['dt'][:] = dt

    # Fill the peaks with random length data
    for p in peaks:
        length = np.random.randint(0, data_length)
        p['length'] = length
        wf = np.random.random(size=data_length)
        p['data'] = wf

        for i in range(length, len(wf)):
            # Set all the out of length data to zero
            wf[i] = 0.

        if len(peaks):
            # Compute sum area
            peaks['area'] = np.sum(peaks['data'], axis=1)

    # Make a copy of peaks to test that they don't remain the same later
    pre_peaks = peaks.copy()
    strax.compute_widths(peaks)

    assert len(pre_peaks) == len(peaks), "Lost peaks"
    if np.sum(peaks['area'] > 0) > 10:
        mess = ("Highly unlikely that from at least 10 positive area peaks "
                "none were able to compute the width")
        assert np.any(peaks['width'] != pre_peaks['width']), mess
