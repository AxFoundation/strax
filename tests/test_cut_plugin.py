from strax import testutils
import strax
import numpy as np
from hypothesis import given, strategies, example, settings

# Initialize. We test both dt time-fields and time time-field
_dtype_name = 'var'
_cut_dtype = ('variable 0', _dtype_name)
full_dt_dtype = [(_cut_dtype, np.float64)] + strax.time_dt_fields
full_time_dtype = [(_cut_dtype, np.float64)] + strax.time_fields


def get_some_array():
    # Either 0 or 1
    take_dt = np.random.choice(2)

    # Stolen from testutils.bounds_to_intervals
    def bounds_to_intervals(bs, dt=1):
        x = np.zeros(len(bs),
                     dtype=full_dt_dtype if take_dt else full_time_dtype)
        x['time'] = [x[0] for x in bs]
        # Remember: exclusive right bound...
        if take_dt:
            x['length'] = [x[1] - x[0] for x in bs]
            x['dt'] = 1
        else:
            x['endtime'] = x['time'] + ([x[1] - x[0] for x in bs]) * dt
        return x

    # Randomly input either of full_dt_dtype or full_time_dtype
    sorted_intervals = testutils.sorted_bounds().map(bounds_to_intervals)
    return sorted_intervals


@given(get_some_array().filter(lambda x: len(x) >= 0),
       strategies.integers(min_value=-10, max_value=10))
@settings(deadline=None)
# Examples for readability
@example(
    input_peaks=np.array(
        [(-11, 0, 1),
         (0, 1, 3),
         (-5, 3, 5),
         (11, 5, 7),
         (7, 7, 9)
         ],
        dtype=[(_cut_dtype, np.float64)] + strax.time_fields),
    cut_threshold=5)
@example(
    input_peaks=np.array(
        [(0, 0, 1, 1),
         (1, 1, 1, 1),
         (5, 2, 2, 1),
         (11, 4, 2, 4)
         ],
        dtype=[(_cut_dtype, np.int16)] + strax.time_dt_fields),
    cut_threshold=-1)
def test_cut_plugin(input_peaks, cut_threshold):
    """
    """
    # Just one chunk will do
    chunks = [input_peaks]
    _dtype = input_peaks.dtype

    class ToBeCut(strax.Plugin):
        """Data to be cut with strax.CutPlugin"""
        depends_on = tuple()
        dtype = _dtype
        provides = 'to_be_cut'
        data_kind = 'to_be_cut'  # match with depends_on below

        def compute(self, chunk_i):
            data = chunks[chunk_i]
            return self.chunk(
                data=data,
                start=(int(data[0]['time']) if len(data)
                       else np.arange(len(chunks))[chunk_i]),
                end=(int(strax.endtime(data[-1])) if len(data)
                     else np.arange(1, len(chunks) + 1)[chunk_i]))

        # Hack to make peak output stop after a few chunks
        def is_ready(self, chunk_i):
            return chunk_i < len(chunks)

        def source_finished(self):
            return True

    class CutSomething(strax.CutPlugin):
        """Minimal working example of CutPlugin"""

        depends_on = ('to_be_cut',)

        def cut_by(self, to_be_cut):
            return to_be_cut[_dtype_name] > cut_threshold

    st = strax.Context(storage=[])
    st.register(ToBeCut)
    st.register(CutSomething)

    result = st.get_array(run_id='some_run',
                          targets=strax.camel_to_snake(CutSomething.__name__))
    correct_answer = np.sum(input_peaks[_dtype_name] > cut_threshold)
    assert len(result) == len(input_peaks), "WTF??"
    assert correct_answer == np.sum(result['cut_something']), (
        "Cut plugin does not give boolean arrays correctly")

    if len(input_peaks):
        assert strax.endtime(input_peaks).max() == \
               strax.endtime(result).max(), "last end time got scrambled"
        assert np.all(input_peaks['time'] ==
                      result['time']), "(start) times got scrambled"
        assert np.all(strax.endtime(input_peaks) ==
                      strax.endtime(result)), "Some end times got scrambled"
