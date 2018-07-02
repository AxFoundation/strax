from . import helpers   # Mocks numba    # noqa

import numpy as np

from hypothesis import given, strategies

import strax


@given(helpers.disjoint_sorted_intervals.filter(lambda x: len(x) > 0),
       strategies.integers(min_value=0, max_value=3))
def test_overlap_plugin(input_peaks, split_i):
    """Counting number of nearby peaks should not depend on the chunking"""
    chunks = np.split(input_peaks, [split_i])
    chunks = [c for c in chunks if len(c)]
    print("\n\nNew run")
    print(strax.endtime(input_peaks),
          [strax.endtime(c) for c in chunks],
          [len(c) for c in chunks])

    class Peaks(strax.Plugin):
        depends_on = tuple()
        dtype = strax.interval_dtype

        def compute(self, chunk_i):
            return chunks[chunk_i]

        def is_ready(self, chunk_i):
            print(f"ready check for {chunk_i}, {len(chunks)},"
                  f" {chunk_i < len(chunks)}")
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
        dtype = [('n_within_window', np.int16)]

        def get_window_size(self):
            return window

        def compute(self, peaks):
            print(f"Compute got {strax.endtime(peaks)}")
            result = dict(
                n_within_window=count_in_window(strax.endtime(peaks)))
            print(f"Result is %s" % result)
            return result

        def iter(self, *args, **kwargs):
            yield from super().iter(*args, **kwargs)

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
