import math
import numpy as np

import strax
from hypothesis import given, settings, example
import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
from strax.testutils import fake_hits


# -----------------------
# Concatenated overlapping hits:
# -----------------------
@given(fake_hits,
       fake_hits,
       st.integers(min_value=0, max_value=10),
       st.integers(min_value=0, max_value=10))
@settings(deadline=None)
def test_concat_overlapping_hits(hits0, hits1, le, re):
    # combining fake hits of the two channels:
    hits1['channel'] = 1
    hits = np.concatenate([hits0, hits1])

    if not len(hits):
        # In case there are no hitlets there is not much to do:
        concat_hits = strax.concat_overlapping_hits(hits, (le, re), (0, 1), 0, float('inf'))
        assert not len(concat_hits), 'Concatenated hits not empty although hits are empty'

    else:
        hits = strax.sort_by_time(hits)

        # Additional offset to time since le > hits['time'].min() does not
        # make sense:
        hits['time'] += 100

        # Now we are ready for the tests:
        # Creating for each channel a dummy array.
        tmax = strax.endtime(hits).max()  # Since dt is one this is the last sample
        tmax += re

        dummy_array = np.zeros((2, tmax), np.int)
        for h in hits:
            # Filling samples with 1 if inside a hit:
            st = h['time'] - le
            et = strax.endtime(h) + re
            dummy_array[h['channel'], st:et] = 1

        # Now we concatenate the hits and check whether their length matches
        # with the total sum of our dummy arrays.
        concat_hits = strax.concat_overlapping_hits(hits, (le, re), (0, 1), 0, float('inf'))

        assert len(concat_hits) <= len(hits), 'Somehow we have more hits than before ?!?'

        for ch in [0, 1]:
            dummy_sum = np.sum(dummy_array[ch])

            # Computing total length of concatenated hits:
            diff = strax.endtime(concat_hits) - concat_hits['time']
            m = concat_hits['channel'] == ch
            concat_sum = np.sum(diff[m])

            assert concat_sum == dummy_sum, f'Total length of concatenated hits deviates from hits for channel {ch}'

            if len(concat_hits[m]) > 1:
                # Checking if new hits do not overlapp or touch anymore:
                mask = strax.endtime(concat_hits[m])[:-1] - concat_hits[m]['time'][1:]
                assert np.all(mask < 0), f'Found two hits within {ch} which are touching or overlapping'


# -----------------------------
# Test for get_hitlets_data.
# This test is done with some predefined
# records.
# -----------------------------

def test_get_hitlets_data():
    dummy_records = [  # Contains Hitlet #:
        [[1, 3, 2, 1, 0, 0], ],  # 0
        [[0, 0, 0, 0, 1, 3],  # 1
         [2, 1, 0, 0, 0, 0]],  #
        [[0, 0, 0, 0, 1, 3],  # 2
         [2, 1, 0, 1, 3, 2], ],  # 3
        [[0, 0, 0, 0, 1, 2],  # 4
         [2, 2, 2, 2, 2, 2],
         [2, 1, 0, 0, 0, 0]],
        [[2, 1, 0, 1, 3, 2]],  # 5, 6
        [[2, 2, 2, 2, 2, 2]]  # 7
    ]

    # Defining the true parameters of the hitlets:
    true_area = [7, 7, 7, 6, 18, 3, 6, 12]
    true_time = [10, 28, 46, 51, 68, 88, 91, 104]
    true_waveform = [[1, 3, 2, 1],
                     [1, 3, 2, 1],
                     [1, 3, 2, 1],
                     [1, 3, 2],
                     [1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
                     [2, 1],
                     [1, 3, 2],
                     [2, 2, 2, 2, 2, 2]
                     ]

    records = _make_fake_records(dummy_records)
    hits = strax.find_hits(records, min_amplitude=2)
    hits = strax.concat_overlapping_hits(hits, (1, 1), (0, 1), 0, float('inf'))
    hitlets = np.zeros(len(hits), strax.hitlet_with_data_dtype(n_samples=np.max(hits['length'])))
    strax.refresh_hit_to_hitlets(hits, hitlets)
    strax.get_hitlets_data(hitlets, records, np.array([1, 1]))

    for i, (a, wf, t) in enumerate(zip(true_area, true_waveform, true_time)):
        h = hitlets[i]
        assert h['area'] == a, f'Hitlet {i} has the wrong area'
        assert np.all(h['data'][:h['length']] == wf), f'Hitlet {i} has the wrong waveform'
        assert h['time'] == t, f'Hitlet {i} has the wrong starttime'


def _make_fake_records(dummy_records):
    """
    Creates some specific records to test get_hitlet_data.
    """
    nfragments = [len(f) for f in dummy_records]
    records = np.zeros(np.sum(nfragments), strax.record_dtype(6))
    records['dt'] = 1
    time_offset = 10  # Need some start time to avoid negative times

    fragment_ind = 0
    for dr, nf in zip(dummy_records, nfragments):
        for ind, f in enumerate(dr):
            r = records[fragment_ind]
            r['time'] = time_offset
            if ind != (nf - 1):
                r['length'] = len(f)
            else:
                r['length'] = len(f) - _count(f)
            r['data'] = f
            r['record_i'] = ind

            if ind == (nf - 1):
                time_offset += r['length'] + 10  # +10 to ensure non-overlap
            else:
                time_offset += r['length']

            fragment_ind += 1

    pnf = 0
    for nf in nfragments:
        records['pulse_length'][pnf:nf + pnf] = np.sum(records['length'][pnf:nf + pnf])
        pnf += nf
    return records


def _count(data):
    """
    Function which returns number of ZLE samples.
    """
    data = data[::-1]
    ZLE = True
    i = 0
    while ZLE:
        if (not data[i] == 0) or (i == len(data)):
            break
        i += 1
    return i


# -----------------------------
# Test for hitlet_properties.
# This test includes the fwxm and
# refresh_hit_to_hitlets.
# -----------------------------
@st.composite
def hits_n_data(draw, strategy):
    hits = draw(strategy)

    data_list = []
    filter_min_area = lambda x: np.sum(x[:length]) >= 0.1
    for i, h in enumerate(hits):
        length = hits[i]['length']
        data = draw(hnp.arrays(
            shape=int(hits['length'].max()),
            dtype=np.float32,
            elements=st.floats(min_value=-2, max_value=10, width=32),
            fill=st.nothing()).filter(filter_min_area))
        data_list.append(data)
    data = np.array(data_list)
    hd = (hits, data)
    return hd


def test_highest_density_region_width():
    """
    Some unit test for the HDR width estimate.
    """
    truth_dict = {0.5: [[2 / 3, 2 + 1 / 3]], 0.8: [0., 4.], 0.9: [-0.25, 4.25]}
    # Some distribution with offset:
    _test_highest_density_region_width(np.array([1, 7, 1, 1, 0]), truth_dict)

    # Same but with offset (zero missing):
    _test_highest_density_region_width(np.array([1, 7, 1, 1]), truth_dict)

    # Two more nasty cases:
    truth_dict = {0.5: [[0, 1]], 0.8: [-0.3, 1.3]}
    _test_highest_density_region_width(np.array([1]), truth_dict)

    truth_dict = {0.5: [[0, 1]], 0.8: [-0.3, 1.3]}
    _test_highest_density_region_width(np.array([1, 0]), truth_dict)


def _test_highest_density_region_width(distribution, truth_dict):
    res = strax.processing.hitlets.highest_density_region_width(distribution,
                                                                np.array(list(truth_dict.keys())),
                                                                fractionl_edges=True)

    for ind, (fraction, truth) in enumerate(truth_dict.items()):
        mes = f'Found wrong edges for {fraction} in {distribution} expected {truth} but got {res[ind]}.'
        assert np.all(np.isclose(truth, res[ind])), mes


@given(hits_n_data=hits_n_data(fake_hits))
@settings(deadline=None)
def test_hitlet_properties(hits_n_data):
    """
    Function which tests refresh_hit_to_hitlets, hitlet_with_data_dtype,
    and hitlet_properties.

    :param hits_n_data:
    :return:
    """
    hits, data = hits_n_data

    hits['time'] += 100
    # Step 1.: Produce fake hits and convert them into hitlets:
    if len(hits) >= 1:
        nsamples = hits['length'].max()
    else:
        nsamples = 2

    hitlets = np.zeros(len(hits), dtype=strax.hitlet_with_data_dtype(nsamples))
    if len(hitlets):
        assert hitlets['data'].shape[1] >= 2, 'Data buffer is not at least 2 samples long.'
    strax.refresh_hit_to_hitlets(hits, hitlets)

    # Testing refresh_hit_to_hitlets for free:
    assert len(hits) == len(hitlets), 'Somehow hitlets and hits have different sizes'
    # Testing interval fields:
    dummy = np.zeros(0, dtype=strax.interval_dtype)
    for name in dummy.dtype.names:
        assert np.all(hitlets[name] == hits[name]), f'The entry of the field {name} did not match between hit and ' \
                                                    f'hitlets '

    # Step 2.: Add to each hit(let) some data
    for ind, d in enumerate(data):
        h = hitlets[ind]
        h['data'][:h['length']] = d[:h['length']]

    # Step 3.: Add np.nan in data but outside of length:
    for h in hitlets:
        if h['length'] < len(h['data']):
            h['data'][-1] = np.nan
            # It is enough to test this for a single hitlet:
            break

    # Step 4.: Compute properties and apply tests:
    strax.hitlet_properties(hitlets)
    for ind, d in enumerate(data):
        h = hitlets[ind]
        d = d[:h['length']]
        pos_max = np.argmax(d)

        # Checking amplitude things:
        assert pos_max == h['time_amplitude'], 'Wrong amplitude position found!'
        assert d[pos_max] == h['amplitude'], 'Wrong amplitude value found!'

        # Checking FHWM and FWTM:
        fractions = [0.1, 0.5]
        for f in fractions:
            # Get field names for the correct test:
            if f == 0.5:
                left = 'left'
                fwxm = 'fwhm'
            else:
                left = 'low_left'
                fwxm = 'fwtm'

            amplitude = np.max(d)
            if np.all(d[0] == d) or np.all(d > amplitude * f):
                # If all samples are either the same or greater than required height FWXM is not defined:
                mes = 'All samples are the same or larger than require height.'
                assert np.isnan(h[left]), mes + f' Left edge for {f} should have been np.nan.'
                assert np.isnan(h[left]), mes + f' FWXM for X={f} should have been np.nan.'
            else:
                le = np.argwhere(d[:pos_max] <= amplitude * f)
                if len(le):
                    le = le[-1, 0]
                    m = d[le + 1] - d[le]
                    le = le + 0.5 + (amplitude * f - d[le]) / m
                else:
                    le = 0

                re = np.argwhere(d[pos_max:] <= amplitude * f)

                if len(re) and re[0, 0] != 0:
                    re = re[0, 0] + pos_max
                    m = d[re] - d[re - 1]
                    re = re + 0.5 + (amplitude * f - d[re]) / m
                else:
                    re = len(d)

                assert math.isclose(le, h[left],
                                    rel_tol=10**-4, abs_tol=10**-4), f'Left edge does not match for fraction {f}'
                assert math.isclose(re - le, h[fwxm], rel_tol=10**-4,
                                    abs_tol=10**-4), f'FWHM does not match for {f}'

    def test_not_defined_get_fhwm():
        # This is a specific unity test for some edge-cases in which the full
        # width half maximum is not defined.
        odd_hitlets = np.zeros(3, dtype=strax.hitlet_with_data_dtype(10))
        odd_hitlets[0]['data'][:5] = [2, 2, 3, 2, 2]
        odd_hitlets[0]['length'] = 5
        odd_hitlets[1]['data'][:2] = [5, 5]
        odd_hitlets[1]['length'] = 2
        odd_hitlets[2]['length'] = 3

        for oh in odd_hitlets:
            res = strax.get_fwxm(oh)
            mes = (f'get_fxhm returned {res} for {oh["data"][:oh["length"]]}!'
                   'However, the FWHM is not defined and the return should be nan!'
                   )
            assert np.all(np.isnan(res)), mes


# ------------------------
# Entropy test
# ------------------------
data_filter = lambda x: (np.sum(x) == 0) or (np.sum(np.abs(x)) >= 0.1)


@given(data=hnp.arrays(np.float32,
                       shape=st.integers(min_value=1, max_value=10),
                       elements=st.floats(min_value=-10, max_value=10, width=32)).filter(data_filter),
       size_template_and_ind_max_template=st.lists(elements=st.integers(min_value=0, max_value=10), min_size=2,
                                                   max_size=2).filter(lambda x: x[0] != x[1]))
@settings(deadline=None)
# Example that failed once
@example(
    data=np.array([7.9956017, 6.6565537, -7.7413940, -2.8149414, -2.8149414,
                   9.9609370, -2.8149414, -2.8149414, -2.8149414, -2.8149414],
                  dtype=np.float32),
    size_template_and_ind_max_template=[0, 1])
def test_conditional_entropy(data, size_template_and_ind_max_template):
    """
    Test for conditional entropy. For the template larger int value defines
    size of the template, smaller int value position of the maximum.
    """

    hitlet = np.zeros(1, dtype=strax.hitlet_with_data_dtype(n_samples=10))
    ind_max_template, size_template = np.sort(size_template_and_ind_max_template)

    # Make dummy hitlet:
    data = data.astype(np.float32)
    len_data = len(data)
    hitlet['data'][0, :len_data] = data[:]
    hitlet['length'][0] = len_data

    # Test 1.: Flat template and no data:
    e1 = strax.conditional_entropy(hitlet, 'flat')[0]
    if np.sum(data):
        d = data
        d = d / np.sum(d)
        m = d > 0

        template = np.ones(np.sum(m), dtype=np.float32)
        template = template / np.sum(template)

        e2 = - np.sum(d[m] * np.log(d[m] / template))
        assert math.isclose(e1, e2, rel_tol=2 * 10**-4,
                            abs_tol=10**-4), f"Test 1.: Entropy function: {e1}, entropy test: {e2}"

        # Test 2.: Arbitrary template:
        template = np.ones(size_template, dtype=np.float32)
        template[ind_max_template] = 2
        template /= np.sum(template)

        # Aligning data in a slightly different way as in the function
        # itself:
        e2 = _align_compute_entropy(d, template)

        e1 = strax.conditional_entropy(hitlet, template)[0]
        assert math.isclose(e1, e2, rel_tol=2 * 10**-4,
                            abs_tol=10**-4), f"Test 2.: Entropy function: {e1}, entropy test: {e2}"

        # Test 3.: Squared waveform:
        # Same as before but this time we square the template and the
        # data.
        template = np.ones(size_template, dtype=np.float32)
        template[ind_max_template] = 2
        template = template * template
        template /= np.sum(template)

        d = data * data
        d = d / np.sum(d)

        e2 = _align_compute_entropy(d, template)

        e1 = strax.conditional_entropy(hitlet, template, square_data=True)[0]
        assert math.isclose(e1, e2, rel_tol=10**-4,
                            abs_tol=10**-4), f"Test 3.: Entropy function: {e1}, entropy test: {e2}"
    else:
        assert np.isnan(e1), f'Hitlet entropy is {e1}, but expected np.nan'


def _align_compute_entropy(data, template):
    ind_max_data = np.argmax(data)
    len_data = len(data)

    ind_max_template = np.argmax(template)
    len_template = len(template)

    # Aligning data in a slightly different way as in the function
    # itself:
    max_to_end = min(len_template - ind_max_template,
                     len_data - ind_max_data)
    start_to_max = min(ind_max_template, ind_max_data)

    template_aligned = template[ind_max_template - start_to_max:ind_max_template + max_to_end]
    data_aligned = data[ind_max_data - start_to_max:ind_max_data + max_to_end]

    m = template_aligned > 0
    m = m & (data_aligned > 0)
    entropy = -np.sum(data_aligned[m] * np.log(data_aligned[m] / template_aligned[m]))
    return entropy
