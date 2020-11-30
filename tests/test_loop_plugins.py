from strax import testutils
import strax
import numpy as np
from hypothesis import given, strategies, example, settings
import tempfile
# Save some selfs some time, let's import from cut-plugin tests some dummy arrays
from .test_cut_plugin import _dtype_name, full_dt_dtype, full_time_dtype, get_some_array


def rechunk_array_to_arrays(array, n: int):
    """Yield successive n-sized chunks from array."""
    for i in range(0, len(array), n):
        yield array[i:i + n]


def drop_random(chunks: list) -> list:
    """
    Drop some of the data in the chunks
    :param chunks: list op numpy arrays to modify. Here we will drop some of the fields randomly
    :return: list of chunks
    """
    res = []
    for chunk in chunks:
        if len(chunk) > 1:
            # We are going to keep this many items in this chunk
            keep_n = np.random.randint(1, len(chunk)+1)
            # These are the indices we will keep (only keep unique ones)
            keep_indices = np.random.randint(0, len(chunk)-1, keep_n)
            keep_indices = np.unique(keep_indices)
            keep_indices.sort()

            # This chunk will now be reduced using only keep_indices
            d = chunk[keep_indices]
            res.append(d)
    return res


def _loop_test_inner(big_data, nchunks, target='added_thing', force_value_error=False):
    """
    Test loop plugins for random data. For this test we are going to
    setup to plugins that will be looped over and combined into a loop
    plugin (depending on the target, this may be a multi output plugin).

    We are going to setup as follows:
     - setup chunks for a big data plugin (where we will loop over later)
     - generate some data with similar chunking called 'small data' this
       we will add to the big data in the loop plugin
    """

    if len(big_data) or force_value_error:
        # Generate some random amount of chunks for the big-data
        big_chunks = list(rechunk_array_to_arrays(big_data, nchunks))
    else:
        # If empty, there is no reason to make multiple empty chunks
        # unless we want to force the ValueError later
        big_chunks = [big_data]

    _dtype = big_data.dtype

    # TODO smarter test. I want to drop some random data from the
    #  small_chunks but this does not work yet. Perhaps related to
    #  https://github.com/AxFoundation/strax/pull/345 (will fix in that
    #  PR)
    # small_chunks = drop_random(big_chunks.copy()) # What I want to do
    small_chunks = big_chunks

    class BigThing(strax.Plugin):
        """Plugin that provides data for looping over"""
        depends_on = tuple()
        dtype = _dtype
        provides = 'big_thing'
        data_kind = 'big_kinda_data'

        def compute(self, chunk_i):
            data = big_chunks[chunk_i]
            chunk = self.chunk(
                data=data,
                start=(
                    int(data[0]['time']) if len(data)
                    else np.arange(len(big_chunks))[chunk_i]),
                end=(
                    int(strax.endtime(data[-1])) if len(data)
                    else np.arange(1, len(big_chunks) + 1)[chunk_i]))
            return chunk

        def is_ready(self, chunk_i):
            # Hack to make peak output stop after a few chunks
            return chunk_i < len(big_chunks)

        def source_finished(self):
            return True

    class SmallThing(strax.CutPlugin):
        """Minimal working example of CutPlugin"""
        depends_on = tuple()
        provides = 'small_thing'
        data_kind = 'small_kinda_data'
        dtype = _dtype

        def compute(self, chunk_i):
            data = small_chunks[chunk_i]
            chunk = self.chunk(
                data=data,
                start=(
                    int(data[0]['time']) if len(data)
                    else np.arange(len(small_chunks))[chunk_i]),
                end=(
                    int(strax.endtime(data[-1])) if len(data)
                    else np.arange(1, len(small_chunks) + 1)[chunk_i]))
            return chunk

        def is_ready(self, chunk_i):
            # Hack to make peak output stop after a few chunks
            return chunk_i < len(small_chunks)

        def source_finished(self):
            return True

    class AddBigToSmall(strax.LoopPlugin):
        """
        Test loop plugin by looping big_thing and adding whatever is in small_thing
        """
        depends_on = 'big_thing', 'small_thing'
        provides = 'added_thing'
        loop_over = 'big_kinda_data'  # Also just test this feature

        def infer_dtype(self):
            # Get the dtype from the dependency
            return self.deps['big_thing'].dtype

        def compute_loop(self, big_kinda_data, small_kinda_data):
            res = {}
            for k in self.dtype.names:
                if k == _dtype_name:
                    res[k] = big_kinda_data[k]
                    for small_bit in small_kinda_data[k]:
                        if np.iterable(res[k]):
                            for i in range(len(res[k])):
                                res[k][i] += small_bit
                        else:
                            res[k] += small_bit
                else:
                    res[k] = big_kinda_data[k]
            return res

    class AddBigToSmallMultiOutput(strax.LoopPlugin):
        depends_on = 'big_thing', 'small_thing'
        provides = 'some_combined_things', 'other_combined_things'
        data_kind = {k: k for k in provides}

        def infer_dtype(self):
            # Get the dtype from the dependency.
            # NB! This should be a dict for the kind of provide arguments
            return {k: self.deps['big_thing'].dtype for k in self.provides}

        def compute_loop(self, big_kinda_data, small_kinda_data):
            res = {}
            for k in self.dtype['some_combined_things'].names:
                if k == _dtype_name:
                    res[k] = big_kinda_data[k]
                    for small_bit in small_kinda_data[k]:
                        if np.iterable(res[k]):
                            for i in range(len(res[k])):
                                res[k][i] += small_bit
                        else:
                            res[k] += small_bit
                else:
                    res[k] = big_kinda_data[k]
            return {k: res for k in self.provides}

    with tempfile.TemporaryDirectory() as temp_dir:
        st = strax.Context(storage=[strax.DataDirectory(temp_dir)])
        st.register((BigThing, SmallThing, AddBigToSmall, AddBigToSmallMultiOutput))
        result = st.get_array(run_id='some_run', targets=target)
        assert np.shape(result) == np.shape(big_data), 'Looping over big_data resulted in a different datasize?!'
        assert np.sum(result[_dtype_name]) >= np.sum(big_data[_dtype_name]), "Result should be at least as big as big_data because we added small_data data"
        assert isinstance(result, np.ndarray), "Result is not ndarray?"


@given(get_some_array().filter(lambda x: len(x) >= 0),
       strategies.integers(min_value=1, max_value=10))
@settings(deadline=None)
@example(
    big_data=np.array(
        [(0, 0, 1, 1),
         (1, 1, 1, 1),
         (5, 2, 2, 1),
         (11, 4, 2, 4)],
        dtype=full_dt_dtype),
    nchunks=2)
def test_loop_plugin(big_data, nchunks):
    """Test the loop plugin for random data"""
    _loop_test_inner(big_data, nchunks)


@given(get_some_array().filter(lambda x: len(x) >= 0),
        strategies.integers(min_value=1, max_value=10))
@settings(deadline=None)
@example(
    big_data=np.array(
        [(0, 0, 1, 1),
         (1, 1, 1, 1),
         (5, 2, 2, 1),
         (11, 4, 2, 4)],
        dtype=full_dt_dtype),
    nchunks=2)
def test_loop_plugin_multi_output(big_data, nchunks,):
    """
    Test the loop plugin for random data where it should give multiple
    outputs
    """
    _loop_test_inner(big_data, nchunks, target='other_combined_things')


@given(get_some_array().filter(lambda x: len(x) == 0),
       strategies.integers(min_value=2, max_value=10))
@settings(deadline=None)
@example(
    big_data=np.array(
        [],
        dtype=full_dt_dtype),
    nchunks=2)
def test_value_error_for_loop_plugin(big_data, nchunks):
    """Make sure that we are are getting the right ValueError"""
    try:
        _loop_test_inner(big_data, nchunks, force_value_error=True)
        raise RuntimeError(
            'did not run into ValueError despite the fact we are having '
            'multiple none-type chunks')
    except ValueError:
        # Good we got the ValueError we wanted
        pass
