import numpy as np

__all__ = ('interval_dtype record_dtype hit_dtype peak_dtype '
           'DIGITAL_SUM_WAVEFORM_CHANNEL').split()

DIGITAL_SUM_WAVEFORM_CHANNEL = -1


# Base dtype for interval-like objects (pulse, peak, hit)
interval_dtype = [
    # Channel number to which this interval applies
    ('channel', np.int16),
    # Sample time resolution in nanoseconds
    ('dt', np.int16),
    # Start time of the interval (ns since unix epoch)
    ('time', np.int64),
    # Length of the interval in samples
    # Don't try to make O(second) long intervals!
    ('length', np.int32),
    # Sub-dtypes MUST contain an area field
    # However, the type varies: float for sum waveforms (area in PE)
    # and int32 for per-channel waveforms (area in ADC x samples)
]


def record_dtype(samples_per_record):
    """Data type for a waveform record.

    Length can be shorter than the number of samples in data,
    this indicates a record with zero-padding at the end.
    """
    return interval_dtype + [
        # Integral in ADC x samples
        ('area', np.int32),
        # Total samples in the PULSE to which this record belongs
        # (excluding zero-padding in the last record)
        # Don't try more than 32 767 samples/pulse...
        ('pulse_length', np.int16),
        # Position of this record in the pulse
        ('record_i', np.int16),
        # Baseline in ADC counts
        # If this field is nonzero, the record has been int-baselined
        # That is, data = int(baseline) - original_data
        ('baseline', np.float32),
        # Level of data reduction applied (strax.Reduction enum)
        ('reduction_level', np.uint8),
        # Waveform data. Note this is defined as a SIGNED integer, so we can
        # still represent negative values after subtracting baselines
        ('data', np.int16, samples_per_record),
    ]


# Data type for a 'hit': a sub-range of a record
hit_dtype = interval_dtype + [
    # Integral in ADC x samples. TODO: Currently not set.
    ('area', np.int32),
    # Start sample of hit in record
    ('left', np.int16),
    # End sample of hit in record - exclusive bound! (like python ranges)
    ('right', np.int16),
    # Record index in which the hit was found
    # I assume nobody will process more than 2 147 483 647 pulses at a time
    ('record_i', np.int32),
]


def peak_dtype(n_channels, n_sum_wv_samples=200, n_widths=11):
    """Data type for peaks - ranges across all channels in a detector
    Remember to set channel to -1 (todo: make enum)
    """
    return interval_dtype + [
        # Sum integral in PE
        ('area', np.float32),
        # Area per channel in ADC * samples
        ('area_per_channel', (np.int32, n_channels)),
        # Number of hits from which this peak was constructed
        # (zero if peak was split afterwards)
        ('n_hits', np.int16),
        # Waveform data in PE/sample
        ('data', np.float32, n_sum_wv_samples),
        # Peak widths (range_area_decile)
        ('width', np.float32, n_widths)
    ]
