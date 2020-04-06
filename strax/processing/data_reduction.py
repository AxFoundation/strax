"""Functions to perform in-place pulse-level data reduction"""
import numpy as np
import numba
from enum import IntEnum

import strax
from strax.processing.pulse_processing import NO_RECORD_LINK, record_links
export, __all__ = strax.exporter()


@export
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


@export
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
        clear_from -= d.record_i.astype(np.int32) * samples_per_record
        clear_from = max(0, clear_from)
        if clear_from < samples_per_record:
            d.data[clear_from:] = 0
        d['reduction_level'] = ReductionLevel.BASELINE_CUT


@export
def cut_outside_hits(records, hits, left_extension=2, right_extension=15):
    """Return records with waveforms zeroed if not within
    left_extension or right_extension of hits.
    These extensions properly account for breaking of pulses into records.

    If you pass an incomplete (e.g. cut) set of records, we will not save
    data around hits found in the removed records, even if this stretches
    into records that you did pass.
    """
    if not len(records):
        return records

    # Create a copy of records with blanked data
    # Even a simple records.copy() is mightily slow in numba,
    # and assignments to struct arrays seem troublesome.
    # The obvious solution:
    #     new_recs = records.copy()
    #     new_recs['data'] = 0
    # is quite slow.
    # Replacing the last = with *= gives a factor 2 speed boost.
    # But ~40% faster still is this:
    meta_fields = [x for x in records.dtype.names
                   if x not in ['data', 'reduction_level']]

    new_recs = np.zeros(len(records), dtype=records.dtype)
    new_recs[meta_fields] = records[meta_fields]
    new_recs['reduction_level'] = ReductionLevel.HITS_ONLY

    _cut_outside_hits(records, hits, new_recs,
                      left_extension, right_extension)

    return new_recs


@numba.jit(nopython=True, nogil=True, cache=True)
def _cut_outside_hits(records, hits, new_recs,
                      left_extension=2, right_extension=15):
    if not len(records):
        return
    samples_per_record = len(records[0]['data'])

    previous_record, next_record = record_links(records)

    for hit_i, h in enumerate(hits):
        rec_i = h['record_i']
        r = records[rec_i]

        # Indices to keep, with 0 at the start of this record
        start_keep = h['left'] - left_extension
        end_keep = h['right'] + right_extension

        # Indices of samples to keep in this record
        (a, b), _ = strax.overlap_indices(0, r['length'],
                                          start_keep, end_keep - start_keep)
        new_recs[rec_i]['data'][a:b] = records[rec_i]['data'][a:b]

        # Keep samples in previous record, if there was one
        if start_keep < 0:
            prev_ri = previous_record[rec_i]
            if prev_ri != NO_RECORD_LINK:
                # Note start_keep is negative, so this keeps the
                # last few samples of the previous record
                a_prev = start_keep
                new_recs[prev_ri]['data'][a_prev:] = \
                    records[prev_ri]['data'][a_prev:]

        # Same for the next record, if there is one
        if end_keep > samples_per_record:
            next_ri = next_record[rec_i]
            if next_ri != NO_RECORD_LINK:
                b_next = end_keep - samples_per_record
                new_recs[next_ri]['data'][:b_next] = \
                    records[next_ri]['data'][:b_next]
