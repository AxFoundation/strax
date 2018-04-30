"""Functions to perform in-place pulse-level data reduction"""
import numpy as np
import numba
from enum import IntEnum

from strax.processing.pulse_processing import NOT_APPLICABLE, record_links
from strax.processing.peak_building import find_peaks
from .general import fully_contained_in
from strax.dtypes import peak_dtype

__all__ = 'ReductionLevel cut_baseline cut_outside_hits ' \
          'replace_with_spike exclude_tails'.split()


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
    WAVEFORM_REPLACED = 3
    # The raw waveform has been deleted, only metadata survives
    METADATA_ONLY = 4


@numba.jit(nopython=True, nogil=True, cache=True)
def cut_baseline(records, n_before=48, n_after=30):
    """"Replace first n_before and last n_after samples of pulses by 0
    """
    # TODO: records.data.shape[1] gives a numba error (file issue?)
    if not len(records):
        return
    samples_per_record = len(records[0]['data'])

    for d_i, d in enumerate(records):
        if d.record_i == 0:
            d.data[:n_before] = 0

        clear_from = d.pulse_length - n_after
        clear_from -= d.record_i * samples_per_record
        clear_from = max(0, clear_from)
        if clear_from < samples_per_record:
            d.data[clear_from:] = 0

    records.reduction_level[:] = ReductionLevel.BASELINE_CUT


@numba.jit(nopython=True, nogil=True, cache=True)
def cut_outside_hits(records, hits, left_extension=2, right_extension=15):
    """Zero record waveforms not within left_extension or right_extension of
    hits.
    These extensions properly account for breaking of pulses into records.

    If you pass an incomplete (e.g. cut) set of records, we will not save
    data around hits found in the removed records, even if this stretches
    into records that you did pass.
    """
    if not len(records):
        return
    samples_per_record = len(records[0]['data'])

    # For every sample, store if we can cut it or not
    can_cut = np.ones((len(records), samples_per_record), dtype=np.bool_)

    previous_record, next_record = record_links(records)

    for hit_i in range(len(hits)):
        h = hits[hit_i]
        rec_i = h['record_i']

        # Keep required samples in current record
        start_keep = h['left'] - left_extension
        end_keep = h['right'] + right_extension
        can_cut[rec_i][max(0, start_keep):
                       min(end_keep, samples_per_record)] = 0

        # Keep samples in previous/next record if applicable
        if start_keep < 0:
            prev_r = previous_record[rec_i]
            if prev_r != NOT_APPLICABLE:
                can_cut[prev_r][start_keep:] = 0
        if end_keep > samples_per_record:
            next_r = next_record[rec_i]
            if next_r != NOT_APPLICABLE:
                can_cut[next_r][:end_keep - samples_per_record] = 0

    # This is actually quite slow. Perhaps the [:] forces a copy?
    # Without it, however, numba complains...
    for i in range(len(can_cut)):
        records[i]['data'][:] *= ~can_cut[i]
    records.reduction_level[:] = ReductionLevel.HITS_ONLY


@numba.jit(nopython=True, nogil=True, cache=True)
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

    records.reduction_level[:] = ReductionLevel.WAVEFORM_REPLACED


# Cannot jit this guy, find_peaks is not a jitted function
def exclude_tails(records, to_pe,
                  min_area=int(2e5),
                  peak_duration=int(1e4),
                  tail_duration=int(1e7),
                  gap_threshold=300):
    """Return records that do not lie fully in tail after a big peak"""
    # Find peaks using the records as "hits". This is rough, but good enough.
    cut = find_peaks(records, to_pe,
                     gap_threshold=gap_threshold,
                     min_area=min_area,
                     result_dtype=peak_dtype(records['channel'].max() + 1),
                     max_duration=peak_duration)
    # Transform these 'peaks' to ranges to cut.
    # We want to cut tails after peaks, not the peaks themselves.
    cut['time'] += peak_duration        # Don't cut the actual peak
    cut['length'] = tail_duration / cut['dt']
    return records[fully_contained_in(records, cut) == -1]
