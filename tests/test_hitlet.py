import math
import numpy as np

import strax
from hypothesis import given, settings
import hypothesis.extra.numpy
import hypothesis.strategies as st
from strax.testutils import fake_hits


# -----------------------
# Concatenated overlapping hits:
# -----------------------
@given(fake_hits,
       fake_hits,
       hypothesis.strategies.integers(min_value=0, max_value=10),
       hypothesis.strategies.integers(min_value=0, max_value=10))
@settings(deadline=None)
def test_concat_overlapping_hits(hits0, hits1, le, re):
    # combining fake hits of the two channels:
    hits1['channel'] = 1
    hits = np.concatenate([hits0, hits1])

    if not len(hits):
        # In case there are no hitlets there is not much to do:
        concat_hits = strax.concat_overlapping_hits(hits, (le, re), (0, 1))
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
        concat_hits = strax.concat_overlapping_hits(hits, (le, re), (0, 1))

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


# ------------------------
# Entropy test
# ------------------------

data_filter = lambda x: (np.sum(x) == 0) or (np.sum(np.abs(x)) >= 0.1)
@given(data=hypothesis.extra.numpy.arrays(np.float32,
                                          shape=hypothesis.strategies.integers(min_value=1, max_value=10),
                                          elements=hypothesis.strategies.floats(min_value=-10, max_value=10,
                                                                                width=32)).filter(data_filter),
       size_template_and_ind_max_template=st.lists(elements=st.integers(min_value=0, max_value=10),
                                                  min_size=2, max_size=2).filter(lambda x: x[0] != x[1])
      )
@settings(deadline=None)
def test_conditional_entropy(data, size_template_and_ind_max_template):
    """
    Test for conditional entropy. For the template larger int value defines
    size of the tempalte, smaller int value position of the maximum.
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
        assert math.isclose(e1, e2, rel_tol=10**-4, abs_tol=10**-4), f"Test 1.: Entropy function: {e1}, entropy test: {e2}"

        # Test 2.: Arbitrary template:
        template = np.ones(size_template, dtype=np.float32)
        template[ind_max_template] = 2
        template /= np.sum(template)

        # Aligning data in a slightly different way as in the function
        # itself:
        e2 = _align_compute_entropy(d, template)

        e1 = strax.conditional_entropy(hitlet, template)[0]
        assert math.isclose(e1, e2, rel_tol=10**-4, abs_tol=10**-4), f"Test 2.: Entropy function: {e1}, entropy test: {e2}"

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
        assert math.isclose(e1, e2, rel_tol=10**-4, abs_tol=10**-4), f"Test 3.: Entropy function: {e1}, entropy test: {e2}"
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
