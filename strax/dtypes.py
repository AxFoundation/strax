"""Fundamental dtypes for use in strax.

Note that if you change the dtype titles (comments), numba will crash if
there is an existing numba cache. Clear __pycache__ and restart.
TODO: file numba issue.
"""
import numpy as np

__all__ = ('interval_dtype record_dtype hit_dtype peak_dtype '
           'DIGITAL_SUM_WAVEFORM_CHANNEL').split()

DIGITAL_SUM_WAVEFORM_CHANNEL = -1


# Base dtype for interval-like objects (pulse, peak, hit)
interval_dtype = [
    (('Channel/PMT number',
        'channel'), np.int16),
    (('Time resolution in ns',
        'dt'), np.int16),
    (('Start time of the interval (ns since unix epoch)',
        'time'), np.int64),
    # Don't try to make O(second) long intervals!
    (('Length of the interval in samples',
        'length'), np.int32),
    # Sub-dtypes MUST contain an area field
    # However, the type varies: float for sum waveforms (area in PE)
    # and int32 for per-channel waveforms (area in ADC x samples)
]


def record_dtype(samples_per_record=110):
    """Data type for a waveform record.

    Length can be shorter than the number of samples in data,
    this indicates a record with zero-padding at the end.
    """
    return interval_dtype + [
        (("Integral in ADC x samples",
            'area'), np.int32),
        # np.int16 is not enough for some PMT flashes...
        (('Length of pulse to which the record belongs (without zero-padding)',
            'pulse_length'), np.int32),
        (('Fragment number in the pulse',
            'record_i'), np.int16),
        (('Baseline in ADC counts. data = int(baseline) - data_orig',
            'baseline'), np.float32),
        (('Level of data reduction applies (strax.ReductionLevel enum)',
            'reduction_level'), np.uint8),
        # Note this is defined as a SIGNED integer, so we can
        # still represent negative values after subtracting baselines
        (('Waveform data in ADC counts above baseline',
            'data'), np.int16, samples_per_record),
    ]


# Data type for a 'hit': a sub-range of a record
hit_dtype = interval_dtype + [
    (("Integral in ADC x samples",
        'area'), np.int32),
    (('Index of sample in record in which hit starts',
        'left'), np.int16),
    (('Index of first sample in record just beyond hit (exclusive bound)',
        'right'), np.int16),
    (('Internal (temporary) index of fragment in which hit was found',
        'record_i'), np.int32),
]


def peak_dtype(n_channels=100, n_sum_wv_samples=200, n_widths=11):
    """Data type for peaks - ranges across all channels in a detector
    Remember to set channel to -1 (todo: make enum)
    """
    return interval_dtype + [
        (('Integral across channels in photoelectrons',
            'area'), np.float32),
        # Area per channel in ADC * samples
        (('Integral per channel in ADX x samples (not PE!)',
            'area_per_channel'), np.int32, n_channels),
        # Number of hits from which this peak was constructed
        # (zero if peak was split afterwards)
        (("Number of hits from which peak was constructed "
          "(currently zero if peak is split afterwards)",
            'n_hits'), np.int16),
        # Waveform data in PE/sample
        (('Waveform data in PE/sample (not PE/ns!)',
            'data'), np.float32, n_sum_wv_samples),
        (('Peak widths in ns: range of central area fraction',
            'width'), np.float32, n_widths)
    ]
