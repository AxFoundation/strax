import math
import numpy as np

import strax
from hypothesis import given, settings
import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import unittest
from strax.testutils import fake_hits


@given(
    fake_hits,
    fake_hits,
    st.integers(min_value=0, max_value=10),
    st.integers(min_value=0, max_value=10),
)
@settings(deadline=None)
def test_concat_overlapping_hits(hits0, hits1, le, re):
    # combining fake hits of the two channels:
    hits1["channel"] = 1
    hits = np.concatenate([hits0, hits1])

    if not len(hits):
        # In case there are no hitlets there is not much to do:
        concat_hits = strax.concat_overlapping_hits(hits, (le, re), (0, 1), 0, float("inf"))
        assert not len(concat_hits), "Concatenated hits not empty although hits are empty"

    else:
        hits = strax.sort_by_time(hits)

        # Additional offset to time since le > hits['time'].min() does not
        # make sense:
        hits["time"] += 100

        # Now we are ready for the tests:
        # Creating for each channel a dummy array.
        tmax = strax.endtime(hits).max()  # Since dt is one this is the last sample
        tmax += re

        dummy_array = np.zeros((2, tmax), np.int64)
        for h in hits:
            # Filling samples with 1 if inside a hit:
            st = h["time"] - le
            et = strax.endtime(h) + re
            dummy_array[h["channel"], st:et] = 1

        # Now we concatenate the hits and check whether their length matches
        # with the total sum of our dummy arrays.
        concat_hits = strax.concat_overlapping_hits(hits, (le, re), (0, 1), 0, float("inf"))

        assert len(concat_hits) <= len(hits), "Somehow we have more hits than before ?!?"

        for ch in [0, 1]:
            dummy_sum = np.sum(dummy_array[ch])

            # Computing total length of concatenated hits:
            diff = strax.endtime(concat_hits) - concat_hits["time"]
            m = concat_hits["channel"] == ch
            concat_sum = np.sum(diff[m])

            assert (
                concat_sum == dummy_sum
            ), f"Total length of concatenated hits deviates from hits for channel {ch}"

            if len(concat_hits[m]) > 1:
                # Checking if new hits do not overlapp or touch anymore:
                mask = strax.endtime(concat_hits[m])[:-1] - concat_hits[m]["time"][1:]
                assert np.all(
                    mask < 0
                ), f"Found two hits within {ch} which are touching or overlapping"


def test_create_hits_from_hitlets_empty_hits():
    hits = np.zeros(0, dtype=strax.hit_dtype)
    hitlets = strax.create_hitlets_from_hits(hits, (1, 1), (0, 1))
    assert len(hitlets) == 0, "Hitlets should be empty"


class TestGetHitletData(unittest.TestCase):
    def setUp(self):
        self.test_data = [1, 3, 2, 1, 0, 0]
        self.test_data_truth = self.test_data[:-2]
        self.records, self.hitlets = self.make_records_and_hitlets([[self.test_data]])

    def make_records_and_hitlets(self, dummy_records):
        records = self._make_fake_records(dummy_records)
        hits = strax.find_hits(records, min_amplitude=2)
        hitlets = strax.create_hitlets_from_hits(hits, (1, 1), (0, 1), 0, float("inf"))
        return records, hitlets

    def test_inputs_are_empty(self):
        hitlets_empty = np.zeros(0, dtype=strax.hitlet_with_data_dtype(2))
        records_empty = np.zeros(0, dtype=strax.record_dtype(10))

        hitlets_result = strax.get_hitlets_data(hitlets_empty, self.records, np.ones(3000))
        assert len(hitlets_result) == 0, "get_hitlet_data returned result for empty hitlets"

        hitlets_result = strax.get_hitlets_data(hitlets_empty, records_empty, np.ones(3000))
        assert len(hitlets_result) == 0, "get_hitlet_data returned result for empty hitlets"

        with self.assertRaises(ValueError):
            strax.get_hitlets_data(self.hitlets, records_empty, np.ones(3000))

    def test_to_pe_wrong_shape(self):
        self.hitlets["channel"] = 2000
        with self.assertRaises(ValueError):
            strax.get_hitlets_data(self.hitlets, self.records, np.ones(10))

    def test_get_hitlets_data_for_single_hitlet(self):
        hitlets = strax.get_hitlets_data(self.hitlets[0], self.records, np.ones(3000))
        self._test_data_is_identical(hitlets, [self.test_data_truth])

    def test_data_field_is_empty(self):
        hitlets = strax.get_hitlets_data(self.hitlets, self.records, np.ones(3000))
        with self.assertRaises(ValueError):
            strax.get_hitlets_data(hitlets, self.records, np.ones(3000))
        self._test_data_is_identical(hitlets, [self.test_data_truth])

    def test_get_hitlets_data_without_data_field(self):
        hitlets_empty = np.zeros(len(self.hitlets), strax.hitlet_dtype())
        strax.copy_to_buffer(self.hitlets, hitlets_empty, "_copy_hitlets_to_hitlets_without_data")

        hitlets = strax.get_hitlets_data(hitlets_empty, self.records, np.ones(3000))
        self._test_data_is_identical(hitlets, [self.test_data_truth])

    def test_to_short_data_field(self):
        hitlets_to_short = np.zeros(len(self.hitlets), dtype=strax.hitlet_with_data_dtype(2))
        strax.copy_to_buffer(self.hitlets, hitlets_to_short, "_refresh_hit_to_hitlet")
        with self.assertRaises(ValueError):
            strax.get_hitlets_data(hitlets_to_short, self.records, np.ones(3000))

    def test_empty_overlap(self):
        records = np.zeros(3, strax.record_dtype(10))

        # Create fake records for which hitlet overlaps with channel 0
        # although hit is in channel 1. See also github.com/AxFoundation/strax/pull/549
        records["channel"] = (0, 1, 1)
        records["length"] = (10, 3, 10)
        records["time"] = (0, 0, 5)
        records["dt"] = 1
        records["data"][-1] = np.ones(10)

        # Assume we extend our hits by 1 sample hence hitlet starts at 4
        hitlet = np.zeros(1, strax.hitlet_with_data_dtype(11))
        hitlet["time"] = 4
        hitlet["dt"] = 1
        hitlet["length"] = 11
        hitlet["channel"] = 1

        hitlet = strax.get_hitlets_data(hitlet, records, np.ones(10))
        assert hitlet["time"] == 5
        assert hitlet["length"] == 10
        assert np.sum(hitlet["data"]) == 10
        assert hitlet["data"][0, 0] == 1

    def test_get_hitlets_data(self):
        dummy_records = [  # Contains Hitlet #:
            [
                [1, 3, 2, 1, 0, 0],
            ],  # 0
            [[0, 0, 0, 0, 1, 3], [2, 1, 0, 0, 0, 0]],  # 1  #
            [
                [0, 0, 0, 0, 1, 3],  # 2
                [2, 1, 0, 1, 3, 2],
            ],  # 3
            [[0, 0, 0, 0, 1, 2], [2, 2, 2, 2, 2, 2], [2, 1, 0, 0, 0, 0]],  # 4
            [[2, 1, 0, 1, 3, 2]],  # 5, 6
            [[2, 2, 2, 2, 2, 2]],  # 7
        ]

        # Defining the true parameters of the hitlets:
        true_area = [7, 7, 7, 6, 18, 3, 6, 12]
        true_time = [10, 28, 46, 51, 68, 88, 91, 104]
        true_waveform = [
            [1, 3, 2, 1],
            [1, 3, 2, 1],
            [1, 3, 2, 1],
            [1, 3, 2],
            [1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
            [2, 1],
            [1, 3, 2],
            [2, 2, 2, 2, 2, 2],
        ]

        records, hitlets = self.make_records_and_hitlets(dummy_records)
        hitlets = strax.get_hitlets_data(hitlets, records, np.ones(2))

        for i, (a, wf, t) in enumerate(zip(true_area, true_waveform, true_time)):
            h = hitlets[i]
            assert h["area"] == a, f"Hitlet {i} has the wrong area"
            assert np.all(h["data"][: h["length"]] == wf), f"Hitlet {i} has the wrong waveform"
            assert h["time"] == t, f"Hitlet {i} has the wrong starttime"

    @staticmethod
    def _test_data_is_identical(hitlets, data):
        for h, d in zip(hitlets, data):
            data_is_identical = np.all(h["data"][: h["length"]] == d)
            assert data_is_identical, "Did not get the correct waveform"

    def _make_fake_records(self, dummy_records):
        """Creates some specific records to test get_hitlet_data."""
        n_fragments = [len(pulse_fragemetns) for pulse_fragemetns in dummy_records]
        records = np.zeros(np.sum(n_fragments), strax.record_dtype(6))
        records["dt"] = 1
        time_offset = 10  # Need some start time to avoid negative times

        fragment_ind = 0
        for dr, number_of_fragements in zip(dummy_records, n_fragments):
            for record_i, waveform in enumerate(dr):
                r = records[fragment_ind]
                r["time"] = time_offset

                is_not_last_fragment = record_i != (number_of_fragements - 1)
                if is_not_last_fragment:
                    r["length"] = len(waveform)
                else:
                    r["length"] = len(waveform) - self._count_zle_samples(waveform)
                r["data"] = waveform
                r["record_i"] = record_i

                is_last_fragment = record_i == (number_of_fragements - 1)
                if is_last_fragment:
                    time_offset += r["length"] + 10  # +10 to ensure non-overlap
                else:
                    time_offset += r["length"]
                fragment_ind += 1

        pulse_offset = 0
        for number_of_fragements in n_fragments:
            pulse_length = np.sum(
                records["length"][pulse_offset : number_of_fragements + pulse_offset]
            )
            records["pulse_length"][
                pulse_offset : number_of_fragements + pulse_offset
            ] = pulse_length
            pulse_offset += number_of_fragements
        return records

    @staticmethod
    def _count_zle_samples(data):
        """Function which returns number of ZLE samples."""
        data = data[::-1]
        ZLE = True
        i = 0
        while ZLE:
            if (not data[i] == 0) or (i == len(data)):
                break
            i += 1
        return i


@st.composite
def hits_n_data(draw, strategy):
    hits = draw(strategy)

    data_list = []
    filter_min_area = lambda x: np.sum(x[:length]) >= 0.1
    for i, h in enumerate(hits):
        length = hits[i]["length"]
        data = draw(
            hnp.arrays(
                shape=int(hits["length"].max()),
                dtype=np.float32,
                elements=st.floats(min_value=-2, max_value=10, width=32),
                fill=st.nothing(),
            ).filter(filter_min_area)
        )
        data_list.append(data)
    data = np.array(data_list)
    hd = (hits, data)
    return hd


def test_highest_density_region_width():
    """Some unit test for the HDR width estimate."""
    truth_dict = {0.5: [[2 / 3, 2 + 1 / 3]], 0.8: [0.0, 4.0], 0.9: [-0.25, 4.25]}
    # Some distribution with offset:
    _test_highest_density_region_width(np.array([1, 7, 1, 1, 0]), truth_dict)

    # Same but with offset (zero missing):
    _test_highest_density_region_width(np.array([1, 7, 1, 1]), truth_dict)

    # Two more nasty cases:
    truth_dict = {0.5: [[0, 1]], 0.8: [-0.3, 1.3]}
    _test_highest_density_region_width(np.array([1]), truth_dict)

    truth_dict = {0.5: [[0, 1]], 0.8: [-0.3, 1.3]}
    _test_highest_density_region_width(np.array([1, 0]), truth_dict)

    # Check that negative data does not raise:
    res = strax.processing.hitlets.highest_density_region_width(
        np.array([0, -1, -2]), np.array([0.5]), fractionl_edges=True
    )
    assert np.all(np.isnan(res)), "For empty data HDR is not defined, should return np.nan!"


def _test_highest_density_region_width(distribution, truth_dict):
    res = strax.processing.hitlets.highest_density_region_width(
        distribution, np.array(list(truth_dict.keys())), fractionl_edges=True
    )

    for ind, (fraction, truth) in enumerate(truth_dict.items()):
        mes = (
            f"Found wrong edges for {fraction} in {distribution} expected {truth} but got"
            f" {res[ind]}."
        )
        assert np.all(np.isclose(truth, res[ind])), mes


@given(hits_n_data=hits_n_data(fake_hits))
@settings(deadline=None)
def test_hitlet_properties(hits_n_data):
    """Function which tests refresh_hit_to_hitlets, hitlet_with_data_dtype, and hitlet_properties.

    :param hits_n_data:
    :return:

    """
    hits, data = hits_n_data

    hits["time"] += 100
    # Step 1.: Produce fake hits and convert them into hitlets:
    nsamples = 0
    if len(hits) >= 1:
        nsamples = hits["length"].max()
    nsamples = np.max((nsamples, 2))

    hitlets = np.zeros(len(hits), dtype=strax.hitlet_with_data_dtype(nsamples))
    if len(hitlets):
        assert hitlets["data"].shape[1] >= 2, "Data buffer is not at least 2 samples long."
    strax.copy_to_buffer(hits, hitlets, "_refresh_hit_to_hitlet_properties_test")

    # Testing refresh_hit_to_hitlets for free:
    assert len(hits) == len(hitlets), "Somehow hitlets and hits have different sizes"
    # Testing interval fields:
    dummy = np.zeros(0, dtype=strax.interval_dtype)
    for name in dummy.dtype.names:
        assert np.all(
            hitlets[name] == hits[name]
        ), f"The entry of the field {name} did not match between hit and hitlets "

    # Step 2.: Add to each hit(let) some data
    for ind, d in enumerate(data):
        h = hitlets[ind]
        h["data"][: h["length"]] = d[: h["length"]]

    # Step 3.: Add np.nan in data but outside of length:
    for h in hitlets:
        if h["length"] < len(h["data"]):
            h["data"][-1] = np.nan
            # It is enough to test this for a single hitlet:
            break

    # Step 4.: Compute properties and apply tests:
    strax.hitlet_properties(hitlets)
    for ind, d in enumerate(data):
        h = hitlets[ind]
        d = d[: h["length"]]
        pos_max = np.argmax(d)

        # Checking amplitude things:
        assert pos_max == h["time_amplitude"], "Wrong amplitude position found!"
        assert d[pos_max] == h["amplitude"], "Wrong amplitude value found!"


# ------------------------
# Entropy test
# ------------------------
data_filter = lambda x: (np.sum(x) == 0) or (np.sum(np.abs(x)) >= 0.1)


@given(
    data=hnp.arrays(
        np.float32,
        shape=st.integers(min_value=1, max_value=10),
        elements=st.floats(min_value=-10, max_value=10, width=32),
    ).filter(data_filter),
    size_template_and_ind_max_template=st.lists(
        elements=st.integers(min_value=0, max_value=10), min_size=2, max_size=2
    ).filter(lambda x: x[0] != x[1]),
)
@settings(deadline=None)
def test_conditional_entropy(data, size_template_and_ind_max_template):
    """Test for conditional entropy.

    For the template larger int value defines size of the template, smaller int value position of
    the maximum.

    """

    hitlet = np.zeros(1, dtype=strax.hitlet_with_data_dtype(n_samples=10))
    ind_max_template, size_template = strax.stablesort(size_template_and_ind_max_template)

    # Make dummy hitlet:
    data = data.astype(np.float32)
    len_data = len(data)
    hitlet["data"][0, :len_data] = data[:]
    hitlet["length"][0] = len_data

    # Test 1.: Flat template and no data:
    e1 = strax.conditional_entropy(hitlet, "flat")[0]

    sum_data = np.sum(data)
    if sum_data:
        if np.abs(sum_data) < 1e-4 * np.ptp(data):
            # Normalizing may cause significant float32 numerical errors.
            # Do not run any tests: this data is unsuitable for testing since
            # slight numpy <-> numba implementation differences could cause
            # failures that are actually harmless.
            return

        d = data
        d = d / np.sum(d)
        m = d > 0

        template = np.ones(np.sum(m), dtype=np.float32)
        template = template / np.sum(template)

        e2 = -np.sum(d[m] * np.log(d[m] / template))
        assert math.isclose(
            e1, e2, rel_tol=2 * 10**-3, abs_tol=10**-3
        ), f"Test 1.: Entropy function: {e1}, entropy test: {e2}"

        # Test 2.: Arbitrary template:
        template = np.ones(size_template, dtype=np.float32)
        template[ind_max_template] = 2
        template /= np.sum(template)

        # Aligning data in a slightly different way as in the function
        # itself:
        e2 = _align_compute_entropy(d, template)

        e1 = strax.conditional_entropy(hitlet, template)[0]
        assert math.isclose(
            e1, e2, rel_tol=2 * 10**-3, abs_tol=10**-3
        ), f"Test 2.: Entropy function: {e1}, entropy test: {e2}"

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
        assert math.isclose(
            e1, e2, rel_tol=10**-3, abs_tol=10**-3
        ), f"Test 3.: Entropy function: {e1}, entropy test: {e2}"
    else:
        assert np.isnan(e1), f"Hitlet entropy is {e1}, but expected np.nan"


def _align_compute_entropy(data, template):
    ind_max_data = np.argmax(data)
    len_data = len(data)

    ind_max_template = np.argmax(template)
    len_template = len(template)

    # Aligning data in a slightly different way as in the function
    # itself:
    max_to_end = min(len_template - ind_max_template, len_data - ind_max_data)
    start_to_max = min(ind_max_template, ind_max_data)

    template_aligned = template[ind_max_template - start_to_max : ind_max_template + max_to_end]
    data_aligned = data[ind_max_data - start_to_max : ind_max_data + max_to_end]

    m = template_aligned > 0
    m = m & (data_aligned > 0)
    entropy = -np.sum(data_aligned[m] * np.log(data_aligned[m] / template_aligned[m]))
    return entropy
