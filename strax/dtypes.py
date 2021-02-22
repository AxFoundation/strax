"""Fundamental dtypes for use in strax.

Note that if you change the dtype titles (comments), numba will crash if
there is an existing numba cache. Clear __pycache__ and restart.
TODO: file numba issue.
"""
import numpy as np
import typing as ty
import numba


__all__ = ('interval_dtype raw_record_dtype record_dtype hit_dtype peak_dtype '
           'DIGITAL_SUM_WAVEFORM_CHANNEL DEFAULT_RECORD_LENGTH '
           'time_fields time_dt_fields hitlet_dtype hitlet_with_data_dtype '
           'copy_to_buffer').split()

DIGITAL_SUM_WAVEFORM_CHANNEL = -1
DEFAULT_RECORD_LENGTH = 110


time_fields = [
    (('Start time since unix epoch [ns]',
     'time'), np.int64),
    (('Exclusive end time since unix epoch [ns]',
     'endtime'), np.int64)]

time_dt_fields = [
    (('Start time since unix epoch [ns]',
      'time'), np.int64),
    # Don't try to make O(second) long intervals!
    (('Length of the interval in samples',
      'length'), np.int32),
    (('Width of one sample [ns]',
      'dt'), np.int16)]

# Base dtype for interval-like objects (pulse, peak, hit)
interval_dtype = time_dt_fields + [
    (('Channel/PMT number',
        'channel'), np.int16)]


def raw_record_dtype(samples_per_record=DEFAULT_RECORD_LENGTH):
    """Data type for a waveform raw_record.

    Length can be shorter than the number of samples in data,
    this indicates a record with zero-padding at the end.
    """
    return interval_dtype + [
        # np.int16 is not enough for some PMT flashes...
        (('Length of pulse to which the record belongs (without zero-padding)',
            'pulse_length'), np.int32),
        (('Fragment number in the pulse',
            'record_i'), np.int16),
        (('Baseline determined by the digitizer (if this is supported)',
            'baseline'), np.int16),
        # Note this is defined as a SIGNED integer, so we can
        # still represent negative values after subtracting baselines
        (('Waveform data in raw ADC counts',
            'data'), np.int16, samples_per_record)]


def record_dtype(samples_per_record=DEFAULT_RECORD_LENGTH):
    """Data type for a waveform record.

    Length can be shorter than the number of samples in data,
    this indicates a record with zero-padding at the end.
    """
    return interval_dtype + [
        # np.int16 is not enough for some PMT flashes...
        (('Length of pulse to which the record belongs (without zero-padding)',
            'pulse_length'), np.int32),
        (('Fragment number in the pulse',
            'record_i'), np.int16),
        (("Integral in ADC counts x samples",
          'area'), np.int32),
        (('Level of data reduction applied (strax.ReductionLevel enum)',
          'reduction_level'), np.uint8),
        (('Baseline in ADC counts. data = int(baseline) - data_orig',
          'baseline'), np.float32),
        (('Baseline RMS in ADC counts. data = baseline - data_orig',
          'baseline_rms'), np.float32),
        (('Multiply data by 2**(this number). Baseline is unaffected.',
          'amplitude_bit_shift'), np.int16),
        (('Waveform data in raw counts above integer part of baseline',
          'data'), np.int16, samples_per_record),
    ]


# Data type for a 'hit': a sub-range of a record
hit_dtype = interval_dtype + [
    (("Integral [ADC x samples]",
        'area'), np.float32),
    (('Index of sample in record in which hit starts',
        'left'), np.int16),
    (('Index of first sample in record just beyond hit (exclusive bound)',
        'right'), np.int16),
    (('For lone hits, index of sample in record where integration starts',
      'left_integration'), np.int16),
    (('For lone hits, index of first sample beyond integration region',
      'right_integration'), np.int16),
    (('Internal (temporary) index of fragment in which hit was found',
        'record_i'), np.int32),
    (('ADC threshold applied in order to find hits',
        'threshold'), np.float32),
    (('Maximum amplitude above baseline [ADC counts]',
        'height'), np.float32),
]


def hitlet_dtype():
    """
    Hitlet dtype same as peaklet or peak dtype but for hit-kind of
    objects.
    """
    dtype = interval_dtype + [
        (('Total hit area in pe',
          'area'), np.float32),
        (('Maximum of the PMT pulse in pe/sample',
          'amplitude'), np.float32),
        (('Position of the Amplitude in ns (minus "time")',
          'time_amplitude'), np.int16),
        (('Hit entropy',
          'entropy'), np.float32),
        (('Width (in ns) of the central 50% area of the hitlet',
          'range_50p_area'), np.float32),
        (('Width (in ns) of the central 80% area of the hitlet',
          'range_80p_area'), np.float32),
        (('Position of the 25% area decile [ns]',
          'left_area'), np.float32),
        (('Position of the 10% area decile [ns]',
          'low_left_area'), np.float32),
        (('Width (in ns) of the highest density region covering a 50% area of the hitlet',
          'range_hdr_50p_area'), np.float32),
        (('Width (in ns) of the highest density region covering a 80% area of the hitlet',
          'range_hdr_80p_area'), np.float32),
        (('Left edge of the 50% highest density region  [ns]',
          'left_hdr'), np.float32),
        (('Left edge of the 80% highest density region  [ns]',
          'low_left_hdr'), np.float32),
        (('FWHM of the PMT pulse [ns]',
          'fwhm'), np.float32),
        (('Left edge of the FWHM [ns] (minus "time")',
          'left'), np.float32),
        (('FWTM of the PMT pulse [ns]', 'fwtm'),
         np.float32),
        (('Left edge of the FWTM [ns] (minus "time")',
          'low_left'), np.float32),]
    return dtype


def hitlet_with_data_dtype(n_samples):
    """
    Hitlet dtype with data field. Required within the plugins to compute
    hitlet properties. 
    
    :param n_samples: Buffer length of the data field. Make sure it can
        hold the longest hitlet.
    :param n_widths: Number of area deciles width.

    Note:
        The data buffer will be at least 2 samples long
    """
    dtype = hitlet_dtype()
    n_samples = max(n_samples, 2)
    additional_fields = [(('Hitlet data in PE/sample with ZLE (only the first length samples are filled)', 'data'),
                           np.float32, n_samples),
                         (('Fragment number in the pulse',
                           'record_i'), np.int32),
                         ]

    return dtype + additional_fields


def peak_dtype(n_channels=100, n_sum_wv_samples=200, n_widths=11):
    """Data type for peaks - ranges across all channels in a detector
    Remember to set channel to -1 (todo: make enum)
    """
    if n_channels == 1:
        raise ValueError("Must have more than one channel")
        # Otherwise array changes shape?? badness ensues
    return interval_dtype + [
        # For peaklets this is likely to be overwritten:
        (('Classification of the peak(let)',
          'type'), np.int8),
        (('Integral across channels [PE]',
          'area'), np.float32),
        (('Integral per channel [PE]',
          'area_per_channel'), np.float32, n_channels),
        (("Number of hits contributing at least one sample to the peak ",
          'n_hits'), np.int32),
        (('Waveform data in PE/sample (not PE/ns!)',
          'data'), np.float32, n_sum_wv_samples),
        (('Peak widths in range of central area fraction [ns]',
          'width'), np.float32, n_widths),
        (('Peak widths: time between nth and 5th area decile [ns]',
          'area_decile_from_midpoint'), np.float32, n_widths),
        (('Does the channel reach ADC saturation?',
          'saturated_channel'), np.int8, n_channels),
        (('Total number of saturated channels',
          'n_saturated_channels'), np.int16),
        (('Hits within tight range of mean',
          'tight_coincidence'), np.int16),
        (('Largest gap between hits inside peak [ns]',
          'max_gap'), np.int32),
        (('Maximum interior goodness of split',
          'max_goodness_of_split'), np.float32),
    ]


def copy_to_buffer(source: np.ndarray,
                   buffer: np.ndarray,
                   func_name: str,
                   field_names: ty.Tuple[str] = None):
    """
    Copy the data from the source to the destination e.g. raw_records to
        records. To this end, we dynamically create the  njitted function
        with the name 'func_name' (should start with "_").

    :param source: array of input
    :param destination: array of buffer to fill with values from input
    :param func_name: how to store the dynamically created function.
        Should start with an _underscore
    :param field_names: dtype names to copy (if none, use all in the
        source)
    """
    if np.shape(source) != np.shape(buffer):
        raise ValueError('Source should be the same length as the buffer')

    if field_names is None:
        # To lazy to specify what to copy, just do all
        field_names = tuple(n for n in source.dtype.names
                            if n in buffer.dtype.names)
    elif not any([n in buffer.dtype.names for n in field_names]):
        raise ValueError('Trying to copy dtypes that are not in the '
                         'destination')

    if not func_name.startswith('_'):
        raise ValueError('Start function with "_"')

    if func_name not in globals():
        # Create a numba function for us
        _create_copy_function(buffer.dtype, field_names, func_name)

    globals()[func_name](source, buffer)


def _create_copy_function(res_dtype, field_names, func_name):
    """Write out a numba-njitted function to copy data"""
    # Cannot cache = True since we are creating the function dynamically
    code = f'''
@numba.njit(nogil=True)
def {func_name}(source, result): 
    for i in range(len(source)):
        s = source[i]
        r = result[i]
'''
    for d in field_names:
        if d not in res_dtype.names:
            raise ValueError('This cannot happen')
        if np.shape(res_dtype[d]):
            # Copy array fields as arrays
            code += f'\n        r["{d}"][:] = s["{d}"][:]'
        else:
            code += f'\n        r["{d}"] = s["{d}"]'
    # pylint: disable=exec-used
    exec(code, globals())
