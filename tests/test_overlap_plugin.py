from strax import testutils

import numpy as np

from hypothesis import given, strategies, example, settings

import strax


@given(testutils.disjoint_sorted_intervals.filter(lambda x: len(x) > 0),
       strategies.integers(min_value=0, max_value=3))
@settings(deadline=None)
# Examples that trigger issue #49
@example(
    input_peaks=np.array(
        [(0, 1, 1, 0), (1, 10, 1, 0), (11, 1, 1, 0)],
        dtype=strax.interval_dtype),
    split_i=2)
@example(
    input_peaks=np.array(
        [(0, 1, 1, 0), (1, 1, 1, 0), (2, 9, 1, 0), (11, 1, 1, 0)],
        dtype=strax.interval_dtype),
    split_i=3)
# Other example that caused failures at some point
@example(
    input_peaks=np.array(
        [(0, 1, 1, 0), (7, 6, 1, 0), (13, 1, 1, 0)],
        dtype=strax.interval_dtype),
    split_i=2)
def test_overlap_plugin(input_peaks, split_i):
    """Counting the number of nearby peaks should not depend on how peaks are
    chunked.
    """
    chunks = np.split(input_peaks, [split_i])
    chunks = [c for c in chunks if not len(c) == 0]

    class Peaks(strax.Plugin):
        depends_on = tuple()
        dtype = strax.interval_dtype

        def compute(self, chunk_i):
            data = chunks[chunk_i]
            return self.chunk(
                data=data,
                start=int(data[0]['time']),
                end=int(strax.endtime(data[-1])))

        # Hack to make peak output stop after a few chunks
        def is_ready(self, chunk_i):
            return chunk_i < len(chunks)

        def source_finished(self):
            return True

    window = 10

    # Note we must apply this to endtime, not time, since
    # peaks straddling the overlap threshold are assigned to the NEXT window.
    # If we used time it would fail on examples with peaks larger than window.
    # In real life, the window should simply be chosen large enough that this
    # is not an issue.
    def count_in_window(ts, w=window):
        # Terribly inefficient algorithm...
        result = np.zeros(len(ts), dtype=np.int16)
        for i, t in enumerate(ts):
            result[i] = ((ts < t + w) & (ts > t - w)).sum()
        return result

    class WithinWindow(strax.OverlapWindowPlugin):
        depends_on = ('peaks',)
        dtype = [('n_within_window', np.int16)] + strax.time_fields

        def get_window_size(self):
            return window

        def compute(self, peaks):
            return dict(
                n_within_window=count_in_window(strax.endtime(peaks)),
                time=peaks['time'][:1],
                endtime=strax.endtime(peaks)[-1:])

    st = strax.Context(storage=[])
    st.register(Peaks)
    st.register(WithinWindow)

    result = st.get_array(run_id='some_run', targets='within_window')
    expected = count_in_window(strax.endtime(input_peaks))

    assert len(expected) == len(input_peaks), "WTF??"
    assert isinstance(result, np.ndarray), "Did not get an array"
    assert len(result) == len(expected), "Result has wrong length"
    np.testing.assert_equal(result['n_within_window'], expected,
                            "Counting went wrong")
