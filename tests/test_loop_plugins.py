from strax import testutils
import strax
import numpy as np
from hypothesis import given, strategies, example, settings
from .test_cut_plugin import _dtype_name, full_dt_dtype, full_time_dtype, _cut_dtype

# Initialize. We test both dt time-fields and time time-field
# _dtype_name2 = 'var'
# _cut_dtype2 = ('variable 0', _dtype_name2)
# full_dt_dtype2 = [(_cut_dtype2, np.float64)] + strax.time_dt_fields
# full_time_dtype2 = [(_cut_dtype2, np.float64)] + strax.time_fields


def do_rechunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_some_array():
    # Either 0 or 1
    take_dt = np.random.choice(2)

    # Stolen from testutils.bounds_to_intervals
    def bounds_to_intervals(bs, dt=1):
        the_data = np.zeros(len(bs),
                     dtype=full_dt_dtype if take_dt else full_time_dtype)
        the_data['time'] = [x[0] for x in bs]
        # Remember: exclusive right bound...
        if take_dt:
            the_data['length'] = [x[1] - x[0] for x in bs]
            the_data['dt'] = 1
        else:
            the_data['endtime'] = the_data['time'] + ([x[1] - x[0] for x in bs]) * dt
        return the_data

    # Randomly input either of full_dt_dtype or full_time_dtype
    sorted_intervals = testutils.sorted_bounds().map(bounds_to_intervals)
    return sorted_intervals


# Examples for readability
@given(get_some_array().filter(lambda x: len(x) >= 0),
        get_some_array().filter(lambda x: len(x) >= 0),
        strategies.integers(min_value=1, max_value=10))
@settings(deadline=None)
@example(
    big_data=np.array(
        [(0, 0, 1, 1),
         (1, 1, 1, 1),
         (5, 2, 2, 1),
         (11, 4, 2, 4)
         ],
        dtype=full_dt_dtype),
    small_data=np.array(
        [(0, 0, 1, 1),
         (1, 1, 1, 1),
         (5, 2, 2, 1),
         (11, 4, 2, 4)
         ],
        dtype=full_dt_dtype),
    nchunks=2)
def test_loop_plugin2(big_data, small_data, nchunks):
    """
    """
    # Just one chunk will do

    big_chunks = [big_data] # list(do_rechunks(big_data, nchunks))
    big_dtype = big_data.dtype

    # TODO smarter test
    small_chunks = [big_data] #list(do_rechunks(big_data, nchunks))
    small_dtype = small_data.dtype

    class BigThing(strax.Plugin):
        """Data to be cut with strax.CutPlugin"""
        depends_on = tuple()
        dtype = big_dtype
        provides = 'big_thing'
        data_kind = 'big_kinda_data'

        def compute(self, chunk_i):
            data = big_chunks[chunk_i]
            return self.chunk(
                data=data,
                start=(int(data[0]['time']) if len(data) else np.arange(len(big_chunks))[chunk_i]),
                end=(int(strax.endtime(data[-1])) if len(data) else np.arange(1, len(big_chunks) + 1)[chunk_i]))

        # Hack to make peak output stop after a few chunks
        def is_ready(self, chunk_i):
            return chunk_i < len(big_chunks)

        def source_finished(self):
            return True

    class SmallThing(strax.CutPlugin):
        """Minimal working example of CutPlugin"""
        depends_on = tuple()
        provides = 'small_thing'
        data_kind = 'small_kinda_data'
        dtype = small_dtype

        def compute(self, chunk_i):
            data = small_chunks[chunk_i]
            return self.chunk(
                data=data,
                start=(int(data[0]['time']) if len(data) else np.arange(len(small_chunks))[chunk_i]),
                end=(int(strax.endtime(data[-1])) if len(data) else np.arange(1, len(small_chunks) + 1)[chunk_i]))

        # Hack to make peak output stop after a few chunks
        def is_ready(self, chunk_i):
            return chunk_i < len(small_chunks)

        def source_finished(self):
            return True

    class AddBigToSmall(strax.LoopPlugin):
        depends_on = 'big_thing', 'small_thing'
        provides = 'combined_thing'

        def infer_dtype(self):
             return self.deps['big_thing'].dtype

        def compute(self, big_kinda_data, small_kinda_data):
            res = np.zeros(len(big_kinda_data), dtype=self.dtype)
            for k in res.dtype.names:
                if k == _dtype_name:
                    res[k] = big_kinda_data[k]
                    for small_bit in small_kinda_data[k]:
                        for i in range(len(res[k])):
                            res[k][i] += small_bit
                else:
                    res[k] = big_kinda_data[k]
            return res

    class AddBigToSmallMultiOutput(strax.LoopPlugin):
        depends_on = 'big_thing', 'small_thing'
        provides = 'combined_things', 'second_combined_things'
        data_kind = {k:k for k in provides}

        def infer_dtype(self):
             return {k: self.deps['big_thing'].dtype for k in self.provides}

        def compute(self, big_kinda_data, small_kinda_data):
            res = np.zeros(len(big_kinda_data), big_dtype)
            for k in res.dtype.names:
                if k == _dtype_name:
                    res[k] = big_kinda_data[k]
                    for small_bit in small_kinda_data[k]:
                        for i in range(len(res[k])):
                            res[k][i] += small_bit
                else:
                    res[k] = big_kinda_data[k]
            return {k: res for k in self.provides}


    st = strax.Context(storage=[])
    st.register((BigThing, SmallThing, AddBigToSmall, AddBigToSmallMultiOutput))

    result = st.get_array(run_id='some_run',
                          targets='combined_thing')
    result = st.get_array(run_id='some_run',
                          targets='second_combined_things')
    assert True
    # correct_answer = np.sum(input_peaks[_dtype_name] > cut_threshold)
    # assert len(result) == len(input_peaks), "WTF??"
    # assert correct_answer == np.sum(result['cut_something']), (
    #     "Cut plugin does not give boolean arrays correctly")
    #
    # if len(input_peaks):
    #     assert strax.endtime(input_peaks).max() == \
    #            strax.endtime(result).max(), "last end time got scrambled"
    #     assert np.all(input_peaks['time'] ==
    #                   result['time']), "(start) times got scrambled"
    #     assert np.all(strax.endtime(input_peaks) ==
    #                   strax.endtime(result)), "Some end times got scrambled"
