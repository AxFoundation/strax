import strax
from strax.testutils import Records, run_id
import tempfile
import numpy as np
from hypothesis import given
import hypothesis.strategies as st
import typing as ty


def _apply_function_to_data(function
                            ) -> ty.Tuple[np.ndarray, np.ndarray]:
    """
    Inner core to test apply function to data
    :param function: some callable function that takes thee positional
        arguments
    :return: records, records with function applied
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        st = strax.Context(storage=strax.DataDirectory(temp_dir,
                                                       deep_scan=True),
                           register=[Records])

        # First load normal records
        records = st.get_array(run_id, 'records')

        # Next update the context and apply
        st.set_context_config({'apply_data_function': function})
        changed_records = st.get_array(run_id, 'records')
    return records, changed_records


def test_apply_pass_to_data():
    """
    What happens if we apply a function that does nothing (well,
        nothing hopefully)
    :return: None
    """

    def nothing(data, r, t):
        return data

    r, r_changed = _apply_function_to_data(nothing)
    assert np.all(r == r_changed)


@given(st.integers(min_value=-10, max_value=10))
def test_apply_ch_shift_to_data(magic_shift: int):
    """
    Apply some magic shift number to the channel field and check the results
    :param magic_shift: some number to check that we can shift the
        channel field with
    :return: None
    """

    def shift_channel(data, r, t):
        """Add a magic number to the channel field in the data"""
        res = data.copy()
        res['channel'] += magic_shift
        return res

    r, r_changed = _apply_function_to_data(shift_channel)
    assert len(r) == len(r_changed)
    assert np.all((r_changed['channel'] - (r['channel'] + magic_shift)) == 0)


def test_apply_drop_data():
    """
    What if we drop random portions of the data, do we get the right results?
    :return: None
    """

    class Drop:
        """Small class to keep track of the number of dropped rows"""

        dropped = 0

        def drop(self, data, r, t):
            """Drop a random portion of the data"""
            # I was too lazy to write a strategy to get the right number
            # of random drops based on the input records.
            keep = np.random.randint(0, 2, len(data)).astype(np.bool_)

            # Keep in mind that we do this on a per chunk basis!
            self.dropped += len(data) - np.sum(keep)
            res = data.copy()[keep]
            return res

    # Init the bookkeeping class
    dropper = Drop()
    r, r_changed = _apply_function_to_data(dropper.drop)

    # The number of records should e
    assert (dropper.dropped == len(r) - len(r_changed))


def test_accumulate():
    """
    Test the st.accumulate function. Should add the results and
        accumulate per chunk. Lets add channels and verify the results
        are correct.
    :return: None
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        st = strax.Context(storage=strax.DataDirectory(temp_dir,
                                                       deep_scan=True),
                           register=[Records])
        channels_from_array = np.sum(st.get_array(run_id, 'records')['channel'])
        channels = st.accumulate(run_id, 'records', fields='channel')['channel']
    assert (channels_from_array == channels)
