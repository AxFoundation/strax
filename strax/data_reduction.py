"""Functions to perform in-place pulse-level data reduction"""
import numba
from enum import IntEnum

__all__ = 'ReductionLevel cut_baseline replace_with_spike'.split()


class ReductionLevel(IntEnum):
    """Identifies what type of data reduction has been used on a record
    """
    # Record not modified
    NO_REDUCTION = 0
    # Samples near pulse start/end were removed
    BASELINE_CUT = 1
    # Samples far from a threshold excursion were removed
    HITS_ONLY = 2
    # The record has been replaced with a simpler waveform
    WAVEFORM_REPLACEMENT = 3


@numba.jit(nopython=True)
def cut_baseline(records, n_before=48, n_after=30):
    """"Replace first n_before and last n_after samples of pulses by 0"""
    # TODO: records.data.shape[1] gives a numba error (file issue?)
    if not len(records):
        return
    samples_per_record = len(records[0]['data'])

    for d_i, d in enumerate(records):
        if d.record_i == 0:
            d.data[:n_before] = 0

        clear_from = d.total_length - n_after
        clear_from -= d.record_i * samples_per_record
        clear_from = max(0, clear_from)
        if clear_from < samples_per_record:
            d.data[clear_from:] = 0

    records.reduction_level[:] = ReductionLevel.BASELINE_CUT


@numba.jit(nopython=True)
def replace_with_spike(records, also_for_multirecord_pulses=False):
    """Replaces the waveform in each record with a spike of the same integral
    :param also_for_multirecord_pulses: if True, does this even if the pulse
    spans multiple records (so you'll get more than one spike...)
    """
    if not len(records):
        return
    samples_per_record = len(records[0]['data'])

    for i, d in enumerate(records):
        if not (d.record_i == 0 or also_for_multirecord_pulses):
            continue
        # What is the center of this record? It's nontrivial since
        # some records have parts that do not represent data at the end
        center = int(min(d.total_length - samples_per_record * d.record_i,
                         samples_per_record) // 2)
        integral = d.data.sum()
        d.data[:] = 0
        d.data[center] = integral

    records.reduction_level[:] = ReductionLevel.WAVEFORM_REPLACEMENT
