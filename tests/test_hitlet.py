import math
import numpy as np

import strax
from hypothesis import given, settings
import hypothesis.extra.numpy
import hypothesis.strategies as st


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
    
    hitlet = np.zeros(1, dtype=strax.hitlet_dtype(n_sample=10))
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
